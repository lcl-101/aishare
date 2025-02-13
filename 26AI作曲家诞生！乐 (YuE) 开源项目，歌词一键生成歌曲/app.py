import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import argparse
import torch
import numpy as np
import json
from omegaconf import OmegaConf
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf

import uuid
from tqdm import tqdm
from einops import rearrange
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import glob
import time
import copy
from collections import Counter
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched
import re
import gradio as gr


parser = argparse.ArgumentParser()
# 模型配置：
# 提示
# 输出 

parser.add_argument("--disable_offload_model", action="store_true", help="如果设置，则模型在第一阶段推理后不会从 GPU 卸载到 CPU.")
parser.add_argument("--cuda_idx", type=int, default=0)
# xcodec 和上采样器的配置
parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml', help='xcodec 配置的 YAML 文件.')
parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='xcodec 检查点的路径.')
parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml', help='Vocos 配置文件的路径.')
parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth', help='Vocos 解码器权重的路径.')
parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth', help='Vocos 解码器权重的路径.')
parser.add_argument('-r', '--rescale', action='store_true', help='重新调整输出以避免削波.')
parser.add_argument("--stage1_model", type=str, default="./checkpoints/YuE-s1-7B-anneal-en-cot")
parser.add_argument("--stage2_model", type=str, default="./checkpoints/YuE-s2-1B-general")
parser.add_argument("--max_new_tokens", type=int, default=3000)
parser.add_argument("--run_n_segments", type=int, default=2)
parser.add_argument("--stage2_batch_size", type=int, default=4)
parser.add_argument("--genre_txt", type=str, default="genre.txt")
parser.add_argument("--lyrics_txt", type=str, default="lyrics.txt")
parser.add_argument("--output_dir", type=str, default="./output")

args = parser.parse_args()

stage1_model_arg = args.stage1_model
stage2_model_arg = args.stage2_model
cuda_idx = args.cuda_idx
# 移除或重命名这些重新赋值语句，以保留 Gradio 小部件：
# max_new_tokens = args.max_new_tokens
# run_n_segments = args.run_n_segments
# stage2_batch_size = args.stage2_batch_size
# output_dir 作为 Gradio 文本框保留
stage1_output_dir = os.path.join(args.output_dir, f"stage1")
stage2_output_dir = stage1_output_dir.replace('stage1', 'stage2')
os.makedirs(stage1_output_dir, exist_ok=True)
os.makedirs(stage2_output_dir, exist_ok=True)

# 加载分词器和模型
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
model = AutoModelForCausalLM.from_pretrained(
    stage1_model_arg, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # 要启用闪电注意力，需要安装 flash-attn
)
# 如果 GPU 可用，传输到设备
model.to(device)
model.eval()

codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(args.resume_path, map_location='cpu')
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()

class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # 转换为单声道
    audio = torch.mean(audio, dim=0, keepdim=True)
    # 如果采样率不符则进行重采样
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

def split_lyrics(lyrics):
    # 使用正则表达式拆分歌词，匹配格式为 [角色]内容，并保证换行后跟随 [ 或文本结尾
    pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    # Convert each tuple to a formatted string.
    return [f"[{speaker}]{text.strip()}" for speaker, text in segments]

# 调用函数并打印结果
stage1_output_set = []
# 提示：
# 流派标签支持纯音乐、流派、情绪、声乐音色和声乐性别
# 需要各种标签

# 指令

random_id = uuid.uuid4()
output_seq = None
# 建议的解码配置
top_p = 0.93
temperature = 1.0
repetition_penalty = 1.2
# 特殊标记
start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
# 格式化文本提示

def run_inference(stage1_model, stage2_model, max_new_tokens, run_n_segments, stage2_batch_size, genre_file, lyrics_file, output_dir):
    global model  # 使用模块级变量
    with open(genre_file.name, 'r', encoding='utf-8') as f:
        genres = f.read().strip()
    with open(lyrics_file.name, 'r', encoding='utf-8') as f:
        lyrics = split_lyrics(f.read())
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics

    run_n_segments = min(run_n_segments+1, len(lyrics))
    for i, p in enumerate(tqdm(prompt_texts[:run_n_segments])):
        section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        guidance_scale = 1.5 if i <=1 else 1.2
        if i==0:
            continue
        if i==1:
            head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
        else:
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device) 
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
        # 如果输出序列超过模型上下文长度，则使用窗口切片
        max_context = 16384-max_new_tokens-1
        if input_ids.shape[-1] > max_context:
            print(f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
            input_ids = input_ids[:, -(max_context):]
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids, 
                max_new_tokens=max_new_tokens, 
                min_new_tokens=100, 
                do_sample=True, 
                top_p=top_p,
                temperature=temperature, 
                repetition_penalty=repetition_penalty, 
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
                guidance_scale=guidance_scale,
                )
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq

    # 保存原始输出并检查完整性
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx)!=len(eoa_idx):
        raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

    vocals = []
    instrumentals = []
    range_begin = 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[1])
        instrumentals.append(instrumentals_ids)
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)
    vocal_save_path = os.path.join(stage1_output_dir, f"cot_{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_vocal_{random_id}".replace('.', '@')+'.npy')
    inst_save_path = os.path.join(stage1_output_dir, f"cot_{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_instrumental_{random_id}".replace('.', '@')+'.npy')
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)
    stage1_output_set.append(vocal_save_path)
    stage1_output_set.append(inst_save_path)


    # offload model
    if not args.disable_offload_model:
        model.cpu()
        del model
        torch.cuda.empty_cache()

    print("Stage 2 inference...")
    model_stage2 = AutoModelForCausalLM.from_pretrained(
        stage2_model_arg, 
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        )
    model_stage2.to(device)
    model_stage2.eval()

    def stage2_generate(model, prompt, batch_size=16):
        codec_ids = codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = codectool.offset_tok_ids(
                        codec_ids, 
                        global_offset=codectool.global_offset, 
                        codebook_size=codectool.codebook_size, 
                        num_codebooks=codectool.num_codebooks, 
                    ).astype(np.int32)
        
        # 根据批次大小或单个输入准备 prompt_ids
        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * 300
                idx_end = (i + 1) * 300
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1
            )
        else:
            # 将二维数组展平成一维
            prompt_ids = np.concatenate([
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids.flatten(),  # Flatten the 2D array to 1D
                np.array([mmtokenizer.stage_2])
            ]).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids).to(device)
        prompt_ids = torch.as_tensor(prompt_ids).to(device)
        len_prompt = prompt_ids.shape[-1]
        
        block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])

        # 教师强制生成循环
        for frames_idx in range(codec_ids.shape[1]):
            cb0 = codec_ids[:, frames_idx:frames_idx+1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            with torch.no_grad():
                stage2_output = model.generate(input_ids=input_ids, 
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=block_list,
                )
            
            assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
            prompt_ids = stage2_output

        # 根据批次大小返回输出
        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output

    def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
        stage2_result = []
        for i in tqdm(range(len(stage1_output_set))):
            output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
            
            if os.path.exists(output_filename):
                print(f'{output_filename} stage2 has done.')
                continue
            
            # Load the prompt
            prompt = np.load(stage1_output_set[i]).astype(np.int32)
            
            # Only accept 6s segments
            output_duration = prompt.shape[-1] // 50 // 6 * 6
            num_batch = output_duration // 6
            
            if num_batch <= batch_size:
                # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
                output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
            else:
                # If num_batch is greater than batch_size, process in chunks of batch_size
                segments = []
                num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

                for seg in range(num_segments):
                    start_idx = seg * batch_size * 300
                    # Ensure the end_idx does not exceed the available length
                    end_idx = min((seg + 1) * batch_size * 300, output_duration*50)  # Adjust the last segment
                    current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                    segment = stage2_generate(
                        model,
                        prompt[:, start_idx:end_idx],
                        batch_size=current_batch_size
                    )
                    segments.append(segment)

                # Concatenate all the segments
                output = np.concatenate(segments, axis=0)
            
            # Process the ending part of the prompt
            if output_duration*50 != prompt.shape[-1]:
                ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
                output = np.concatenate([output, ending], axis=0)
            output = codectool_stage2.ids2npy(output)

            # 修正无效编码（一个粗糙的解决方案，可能影响音频质量）
            fixed_output = copy.deepcopy(output)
            for i, line in enumerate(output):
                for j, element in enumerate(line):
                    if element < 0 or element > 1023:
                        counter = Counter(line)
                        most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                        fixed_output[i, j] = most_frequant
            # save output
            np.save(output_filename, fixed_output)
            stage2_result.append(output_filename)
        return stage2_result

    stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=stage2_batch_size)
    print(stage2_result)
    print('Stage 2 DONE.\n')
    # convert audio tokens to audio
    def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        limit = 0.99
        max_val = wav.abs().max()
        wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
        torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
    # reconstruct tracks
    recons_output_dir = os.path.join(output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, 'mix')
    os.makedirs(recons_mix_dir, exist_ok=True)
    tracks = []
    for npy in stage2_result:
        codec_result = np.load(npy)
        decodec_rlt=[]
        with torch.no_grad():
            decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        decodec_rlt.append(torch.as_tensor(decoded_waveform))
        decodec_rlt = torch.cat(decodec_rlt, dim=-1)
        save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
        tracks.append(save_path)
        save_audio(decodec_rlt, save_path, 16000)
    # mix tracks
    for inst_path in tracks:
        try:
            if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
                and 'instrumental' in inst_path:
                # find pair
                vocal_path = inst_path.replace('instrumental', 'vocal')
                if not os.path.exists(vocal_path):
                    continue
                # mix
                recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('instrumental', 'mixed'))
                vocal_stem, sr = sf.read(inst_path)
                instrumental_stem, _ = sf.read(vocal_path)
                mix_stem = (vocal_stem + instrumental_stem) / 1
                sf.write(recons_mix, mix_stem, sr)
        except Exception as e:
            print(e)

    # vocoder to upsample audios
    vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
    vocoder_output_dir = os.path.join(output_dir, 'vocoder')
    vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
    vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)
    for npy in stage2_result:
        if 'instrumental' in npy:
            # Process instrumental
            instrumental_output = process_audio(
                npy,
                os.path.join(vocoder_stems_dir, 'instrumental.mp3'),
                args.rescale,
                args,
                inst_decoder,
                codec_model
            )
        else:
            # Process vocal
            vocal_output = process_audio(
                npy,
                os.path.join(vocoder_stems_dir, 'vocal.mp3'),
                args.rescale,
                args,
                vocal_decoder,
                codec_model
            )
    # mix tracks
    try:
        mix_output = instrumental_output + vocal_output
        vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
        save_audio(mix_output, vocoder_mix, 44100, args.rescale)
        print(f"Created mix: {vocoder_mix}")
    except RuntimeError as e:
        print(e)
        print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

    # Post process
    replace_low_freq_with_energy_matched(
        a_file=recons_mix,     # 16kHz
        b_file=vocoder_mix,     # 48kHz
        c_file=os.path.join(output_dir, os.path.basename(recons_mix)),
        cutoff_freq=5500.0
    )
    # 返回生成的混音文件路径供预览使用
    return vocoder_mix

# 新增预览函数
def preview_file(file_obj):
    if file_obj is None:
        return ""
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Column():
            stage1_model = gr.Textbox(
                lines=1,
                placeholder="checkpoints/YuE-s1-7B-anneal-en-cot",
                label="第一阶段模型"
            )
            stage2_model = gr.Textbox(
                lines=1,
                placeholder="checkpoints/YuE-s2-1B-general",
                label="第二阶段模型"
            )
            max_new_tokens = gr.Slider(
                minimum=1, maximum=5000, value=3000,
                label="最大新生成令牌数量"
            )
            run_n_segments = gr.Slider(
                minimum=1, maximum=10, value=2,
                label="运行段数"
            )
            stage2_batch_size = gr.Slider(
                minimum=1, maximum=16, value=4,
                label="第二阶段批次大小"
            )
            # 修改后的文件控件及预览控件
            genre_file = gr.File(label="流派文本文档", file_types=[".txt"])
            genre_preview = gr.Textbox(label="流派预览", interactive=False)
            lyrics_file = gr.File(label="歌词文本文档", file_types=[".txt"])
            lyrics_preview = gr.Textbox(label="歌词预览", interactive=False)
            output_dir = gr.Textbox(
                lines=1,
                placeholder="./output",
                label="输出目录"
            )
            # 新增生成音频的预览和播放控件
            generated_audio = gr.Audio(label="生成音频预览")
        # 文件上传后动态更新预览
        genre_file.change(fn=preview_file, inputs=genre_file, outputs=genre_preview)
        lyrics_file.change(fn=preview_file, inputs=lyrics_file, outputs=lyrics_preview)
        run_inference_button = gr.Button("开始推理")
        run_inference_button.click(
            fn=run_inference,
            inputs=[
                stage1_model, stage2_model, max_new_tokens,
                run_n_segments, stage2_batch_size,
                genre_file, lyrics_file, output_dir
            ],
            outputs=generated_audio
        )
        demo.launch(server_name='0.0.0.0')
