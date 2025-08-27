#!/usr/bin/env python3

import os
import re
import sys
import tempfile
import traceback
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
import gradio as gr
from funasr import AutoModel
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 加载模型
model_dir = "checkpoints/SenseVoiceSmall"
try:
    auto_model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print("AutoModel loaded successfully")
except Exception as e:
    print(f"Error loading AutoModel: {e}")
    auto_model = None

try:
    sense_model, sense_kwargs = SenseVoiceSmall.from_pretrained(
        "checkpoints/SenseVoiceSmall", 
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    sense_model.eval()
    print("SenseVoiceSmall model loaded successfully")
except Exception as e:
    print(f"Error loading SenseVoiceSmall: {e}")
    sense_model = None
    sense_kwargs = None

def format_str_v3(s):
    """简化的文本格式化"""
    s = s.replace(" <|", "<|")
    s = s.replace("|> ", "|>")
    
    def remove_and_replace(text, pattern, replacement=""):
        return re.sub(pattern, replacement, text)
    
    # 移除特殊标记
    s = remove_and_replace(s, r'<\|[^|]*\|>')
    s = remove_and_replace(s, r'<\|[^|]*$')
    s = remove_and_replace(s, r'^[^|]*\|>')
    
    # 清理多余空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def format_timestamp(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def generate_subtitle_with_timestamp(text, timestamp_list):
    """基于文本和时间戳生成SRT字幕"""
    if not timestamp_list:
        return "No timestamp information available"
    
    print(f"Generating subtitle for text: {text[:50]}...")
    print(f"Timestamp list length: {len(timestamp_list)}")
    print(f"First timestamp sample: {timestamp_list[0] if timestamp_list else 'None'}")
    
    # 按标点符号分割文本，每个标点符号作为一个字幕行的结束
    # 分割时保留标点符号
    segments = re.split(r'([。！？；，、：\.\!\?\;\,\:])', text)
    
    # 重新组合，每个片段以标点符号结尾作为一行字幕
    combined_sentences = []
    i = 0
    while i < len(segments):
        segment = segments[i].strip()
        if not segment:
            i += 1
            continue
            
        # 如果下一个是标点符号，则组合
        if i + 1 < len(segments) and segments[i + 1] in '。！？；，、：.!?;,:':
            combined_sentences.append(segment + segments[i + 1])
            i += 2
        else:
            # 如果这个片段本身就包含标点符号或者是最后一个片段
            if segment:
                combined_sentences.append(segment)
            i += 1
    
    # 如果没有分割结果，使用原文
    if not combined_sentences:
        combined_sentences = [text]
    
    # 创建字符到时间的映射
    char_to_time = {}
    char_index = 0
    
    try:
        raw_times = []
        for token_info in timestamp_list:
            if isinstance(token_info, (list, tuple)) and len(token_info) >= 3:
                token, ts_left, ts_right = token_info[0], token_info[1], token_info[2]

                # SenseVoiceSmall 已经输出为秒 (参见 model.py 中 (_start*60-30)/1000)
                # 之前误乘 0.02 导致时间被压缩约 50 倍，这里直接使用原值。
                try:
                    start_time = float(ts_left)
                    end_time = float(ts_right)
                except Exception:
                    continue

                # 处理可能的毫秒单位（如果大于 1000 认为是毫秒）
                if start_time > 1000 or end_time > 1000:
                    start_time /= 1000.0
                    end_time /= 1000.0

                # 跳过异常时间
                if end_time < start_time:
                    continue

                raw_times.append(end_time)

                token_str = str(token) if token is not None else ""
                token_len = len(token_str)
                if token_len == 0:
                    continue

                for j in range(token_len):
                    pos = char_index + j
                    if pos < len(text):
                        char_to_time[pos] = {'start': start_time, 'end': end_time}
                char_index += token_len

        if raw_times:
            max_time = max(raw_times)
            min_time = min(raw_times)
            if max_time < 1.0 and len(raw_times) > 5:
                # 仍然异常短，发出调试提示
                print(f"[WARN] Max timestamp {max_time:.3f}s looks too small; check upstream alignment logic.")
            else:
                print(f"Timestamp span: {min_time:.3f}s -> {max_time:.3f}s (duration {max_time-min_time:.3f}s)")
                
    except Exception as e:
        print(f"Error processing timestamps: {e}")
    
    # 生成字幕
    subtitle_lines = []
    subtitle_number = 1
    text_position = 0
    
    for sentence in combined_sentences:
        if not sentence.strip():
            continue
            
        # 找到句子在原文中的位置
        sentence_start_pos = text.find(sentence, text_position)
        if sentence_start_pos == -1:
            sentence_start_pos = text_position
        
        sentence_end_pos = sentence_start_pos + len(sentence)
        text_position = sentence_end_pos
        
        # 获取时间信息
        sentence_start = None
        sentence_end = None
        
        for pos in range(sentence_start_pos, sentence_end_pos):
            if pos in char_to_time:
                if sentence_start is None:
                    sentence_start = char_to_time[pos]['start']
                sentence_end = char_to_time[pos]['end']
        
        # 如果没有找到时间信息，使用邻近字符时间推断
        if sentence_start is None:
            # 尝试从 sentence_start_pos 往前找最近时间
            back_pos = sentence_start_pos - 1
            while back_pos >= 0 and back_pos not in char_to_time:
                back_pos -= 1
            forward_pos = sentence_end_pos
            while forward_pos < len(text) and forward_pos not in char_to_time:
                forward_pos += 1

            if back_pos in char_to_time and forward_pos in char_to_time:
                # 夹在有时间的两段之间，线性插值
                left_t = char_to_time[back_pos]['end']
                right_t = char_to_time[forward_pos]['start']
                sentence_start = left_t + (right_t - left_t) * 0.25
                sentence_end = left_t + (right_t - left_t) * 0.75
            elif back_pos in char_to_time:
                base_t = char_to_time[back_pos]['end']
                sentence_start = base_t + 0.05
                sentence_end = base_t + 1.0
            elif forward_pos in char_to_time:
                base_t = char_to_time[forward_pos]['start']
                sentence_start = max(base_t - 1.0, 0)
                sentence_end = base_t - 0.05
            else:
                # 完全缺失，退回到基于序号的估计，但缩放为 1 秒
                sentence_start = (subtitle_number - 1) * 1.0
                sentence_end = sentence_start + 1.0

        # 确保最小持续时间
        if sentence_end - sentence_start < 0.15:
            sentence_end = sentence_start + 0.15
        
        # 格式化字幕
        start_time_str = format_timestamp(sentence_start)
        end_time_str = format_timestamp(sentence_end)
        
        # 去除所有标点符号（包括中文和英文标点）
        clean_sentence = re.sub(r'[。！？；，、：""''（）\(\)\[\]【】《》〈〉\.\!\?\;\,\:\"\'\-\—\…\~]', '', sentence).strip()
        
        subtitle_lines.append(f"{subtitle_number}")
        subtitle_lines.append(f"{start_time_str} --> {end_time_str}")
        subtitle_lines.append(clean_sentence)
        subtitle_lines.append("")  # 空行分隔
        
        subtitle_number += 1
    
    return "\n".join(subtitle_lines)

def model_inference(input_wav, language, enable_timestamp=False):
    """主要的推理函数"""
    print(f"Starting inference: language={language}, enable_timestamp={enable_timestamp}")
    
    # 预处理音频
    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
    
    # 基本转录
    if auto_model is None:
        return "Error: AutoModel not loaded", ""
    
    try:
        result = auto_model.generate(
            input=input_wav,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True
        )
        
        if not result or len(result) == 0:
            return "Error: No transcription result", ""
        
        text = result[0]["text"]
        formatted_text = format_str_v3(text)
        print(f"Basic transcription: {formatted_text}")
        
    except Exception as e:
        print(f"Error in basic transcription: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", ""
    
    # 时间戳处理
    subtitle_text = ""
    if enable_timestamp:
        if sense_model is None:
            subtitle_text = "Error: SenseVoiceSmall model not loaded"
        else:
            try:
                print("Generating timestamps with SenseVoiceSmall...")
                
                # 创建临时音频文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    wavfile.write(tmp_path, 16000, (input_wav * 32767).astype(np.int16))
                
                # 使用SenseVoiceSmall进行时间戳推理
                sense_result = sense_model.inference(
                    data_in=tmp_path,
                    language=language,
                    use_itn=True,
                    ban_emo_unk=False,
                    output_timestamp=True,
                    **sense_kwargs
                )
                
                print(f"SenseVoice result type: {type(sense_result)}")
                
                if sense_result and len(sense_result) > 0:
                    # 解析结果结构
                    if isinstance(sense_result[0], list) and len(sense_result[0]) > 0:
                        sense_data = sense_result[0][0]
                    else:
                        sense_data = sense_result[0]
                    
                    if isinstance(sense_data, dict) and 'timestamp' in sense_data:
                        timestamp_info = sense_data['timestamp']
                        sense_text = sense_data.get('text', formatted_text)
                        
                        # 处理文本格式
                        sense_text = rich_transcription_postprocess(sense_text)
                        
                        print(f"Found {len(timestamp_info)} timestamp entries")
                        
                        # 生成字幕
                        subtitle_text = generate_subtitle_with_timestamp(sense_text, timestamp_info)
                    else:
                        subtitle_text = "No timestamp information in result"
                else:
                    subtitle_text = "No valid result from SenseVoice model"
                
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"Error generating timestamps: {e}")
                traceback.print_exc()
                subtitle_text = f"Timestamp error: {str(e)}"
    
    return formatted_text, subtitle_text

def launch():
    """启动Gradio界面"""
    # 音频示例
    audio_examples = [
        ["checkpoints/SenseVoiceSmall/example/en.mp3", "en", False],
        ["checkpoints/SenseVoiceSmall/example/zh.mp3", "zh", False],
        ["checkpoints/SenseVoiceSmall/example/ja.mp3", "ja", False],
    ]
    
    # 创建Gradio界面
    with gr.Blocks(title="SenseVoice") as demo:
        gr.Markdown("# SenseVoice Speech Recognition with Timestamps")
        
        with gr.Row():
            with gr.Column():
                # 输入组件
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Audio Input"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=["auto", "zh", "en", "yue", "ja", "ko"],
                    value="auto",
                    label="Language"
                )
                
                timestamp_checkbox = gr.Checkbox(
                    label="Enable Timestamps/Subtitles",
                    value=False
                )
                
                submit_button = gr.Button("Transcribe", variant="primary")
            
            with gr.Column():
                # 输出组件
                transcription_output = gr.Textbox(
                    label="Transcription",
                    placeholder="Transcription will appear here...",
                    lines=8
                )
                
                subtitle_output = gr.Textbox(
                    label="Subtitles (SRT Format)",
                    placeholder="Subtitles will appear here when enabled...",
                    lines=8
                )
        
        # 连接事件
        submit_button.click(
            fn=model_inference,
            inputs=[audio_input, language_dropdown, timestamp_checkbox],
            outputs=[transcription_output, subtitle_output]
        )
        
        # 添加示例
        gr.Examples(
            examples=audio_examples,
            inputs=[audio_input, language_dropdown, timestamp_checkbox],
            outputs=[transcription_output, subtitle_output],
            fn=model_inference,
            cache_examples=False
        )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    launch()
