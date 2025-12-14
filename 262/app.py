# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GLM-TTS Web Application
独立的 Gradio Web 界面，使用本地 checkpoints/GLM-TTS 目录下的模型
"""

import gradio as gr
import torch
import numpy as np
import logging
import os
import gc
from functools import partial

from transformers import AutoTokenizer, LlamaForCausalLM, WhisperFeatureExtractor

from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from llm.glmtts import GLMTTS
from utils.audio import mel_spectrogram
from utils.whisper_models.configuration_whisper import WhisperVQConfig
from utils.whisper_models.modeling_whisper import WhisperVQEncoder
from utils import seed_util
from hyperpyyaml import load_hyperpyyaml
import glob
import safetensors
import pathlib

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型根目录 - 使用本地 checkpoints/GLM-TTS 目录
MODEL_ROOT = os.path.join(CURRENT_DIR, "checkpoints", "GLM-TTS")

# 各模型路径
SPEECH_TOKENIZER_PATH = os.path.join(MODEL_ROOT, "speech_tokenizer")
LLM_PATH = os.path.join(MODEL_ROOT, "llm")
FLOW_CKPT_PATH = os.path.join(MODEL_ROOT, "flow", "flow.pt")
FLOW_CONFIG_PATH = os.path.join(MODEL_ROOT, "flow", "config.yaml")
VOCOS_CKPT_PATH = os.path.join(MODEL_ROOT, "vocos2d", "generator_jit.ckpt")
HIFT_CKPT_PATH = os.path.join(MODEL_ROOT, "hift", "hift.pt")
TOKENIZER_PATH = os.path.join(MODEL_ROOT, "vq32k-phoneme-tokenizer")
FRONTEND_DIR = os.path.join(CURRENT_DIR, "frontend")

# LLM 序列长度限制
MAX_LLM_SEQ_INP_LEN = 750


# --- Token2Wav 类（使用自定义路径）---

class Token2Wav:
    """Token 到波形转换器，使用自定义模型路径"""
    def __init__(self, flow, sample_rate: int = 24000, device: str = "cuda"):
        self.device = device
        self.flow = flow
        self.input_frame_rate = flow.input_frame_rate

        if sample_rate == 32000:
            self.hop_size = 640
            self.sample_rate = 32000
            self.vocoder = load_vocos_jit(device)
        elif sample_rate == 24000:
            self.hop_size = 480
            self.sample_rate = 24000
            self.vocoder = load_hift(device)
        else:
            raise ValueError(f"Unsupported sample_rate: {sample_rate}")
    
    def token2wav_with_cache(self,
                             token_bt,
                             n_timesteps: int = 10,
                             prompt_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
                             prompt_feat: torch.Tensor = torch.zeros(1, 0, 80),
                             embedding: torch.Tensor = torch.zeros(1, 192),
    ):
        if isinstance(token_bt, (list, np.ndarray)):
            token_bt = torch.tensor(token_bt, dtype=torch.long)[None]
        elif not isinstance(token_bt, torch.Tensor):
            raise ValueError(f"Unsupported token_bt type: {type(token_bt)}")

        assert prompt_token.shape[1] != 0 and prompt_feat.shape[1] != 0
        mel, _ = self.flow.inference_with_cache(
            token=token_bt.to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            embedding=embedding.to(self.device),
            n_timesteps=n_timesteps,
        )
        
        wav = self.vocoder(mel)

        return wav, mel


# --- 模型加载工具函数 ---

def load_quantize_encoder(model_path):
    """加载量化编码器"""
    logging.info(f'Loading quantize encoder from {model_path}...')
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


def load_speech_tokenizer(model_path):
    """加载语音 tokenizer"""
    model = load_quantize_encoder(model_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    return model, feature_extractor


def load_flow_model(flow_ckpt_path, config_path, device):
    """加载 Flow 模型"""
    logging.info(f'Loading flow model from {flow_ckpt_path}...')
    with open(config_path, 'r') as f:
        scratch_configs = load_hyperpyyaml(f)
        flow = scratch_configs['flow']

    tmp = torch.load(flow_ckpt_path, map_location=device)
    if isinstance(tmp, dict):
        flow.load_state_dict(tmp["model"])
    else:
        flow.load_state_dict(tmp)

    flow.to(device)
    flow.eval()
    return flow


def load_vocos_jit(device="cuda"):
    """加载 Vocos JIT vocoder (32kHz)"""
    from utils.vocos_util import Vocos2DInference
    logging.info(f"Loading Vocos JIT model from {VOCOS_CKPT_PATH}...")
    return Vocos2DInference(VOCOS_CKPT_PATH, device=device)


def load_hift(device="cuda"):
    """加载 HiFT vocoder (24kHz)"""
    from utils.hift_util import HiFTInference
    logging.info(f"Loading HiFT model from {HIFT_CKPT_PATH}...")
    return HiFTInference(HIFT_CKPT_PATH, device=device)


def get_special_token_ids(tokenize_fn):
    """获取特殊 token IDs"""
    _special_token_ids = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }

    special_token_ids = {}
    endoftext_id = tokenize_fn("<|endoftext|>")[0]
    
    for k, v in _special_token_ids.items():
        __ids = tokenize_fn(v)
        if len(__ids) != 1:
            raise AssertionError(f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}")
        if __ids[0] < endoftext_id:
            raise AssertionError(f"Token '{k}' ({v}) ID {__ids[0]} is smaller than endoftext ID {endoftext_id}")
        special_token_ids[k] = __ids[0]

    return special_token_ids


def load_frontends(speech_tokenizer, sample_rate=24000, use_phoneme=False):
    """加载前端处理模块"""
    if sample_rate == 32000:
        feat_extractor = partial(
            mel_spectrogram, 
            sampling_rate=sample_rate, 
            hop_size=640, 
            n_fft=2560, 
            num_mels=80, 
            win_size=2560, 
            fmin=0, 
            fmax=8000, 
            center=False
        )
        logging.info("Configured for 32kHz frontend.")
    elif sample_rate == 24000:
        feat_extractor = partial(
            mel_spectrogram, 
            sampling_rate=sample_rate, 
            hop_size=480, 
            n_fft=1920, 
            num_mels=80, 
            win_size=1920, 
            fmin=0, 
            fmax=8000, 
            center=False
        )
        logging.info("Configured for 24kHz frontend.")
    else:
        raise ValueError(f"Unsupported sampling_rate: {sample_rate}")

    glm_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(FRONTEND_DIR, "campplus.onnx"),
        os.path.join(FRONTEND_DIR, "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend


# --- 全局模型缓存 ---
MODEL_CACHE = {
    "loaded": False,
    "sample_rate": None,
    "components": None
}


def load_models(use_phoneme=False, sample_rate=24000):
    """加载所有模型"""
    logging.info(f"Loading models with sample_rate={sample_rate}...")
    
    # 加载 Speech Tokenizer
    _model, _feature_extractor = load_speech_tokenizer(SPEECH_TOKENIZER_PATH)
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

    # 加载前端
    frontend, text_frontend = load_frontends(speech_tokenizer, sample_rate=sample_rate, use_phoneme=use_phoneme)

    # 加载 LLM
    logging.info(f"Loading LLM from {LLM_PATH}...")
    llm = GLMTTS(
        llama_cfg_path=os.path.join(LLM_PATH, "config.json"), 
        mode="PRETRAIN"
    )
    llm.llama = LlamaForCausalLM.from_pretrained(LLM_PATH, torch_dtype=torch.float32).to(DEVICE)
    llm.llama_embedding = llm.llama.model.embed_tokens

    special_token_ids = get_special_token_ids(frontend.tokenize_fn)
    llm.set_runtime_vars(special_token_ids=special_token_ids)

    # 加载 Flow 模型
    flow = load_flow_model(FLOW_CKPT_PATH, FLOW_CONFIG_PATH, DEVICE)

    # 创建 Token2Wav 转换器（使用自定义类）
    token2wav = Token2Wav(flow, sample_rate=sample_rate, device=DEVICE)

    logging.info("All models loaded successfully.")
    return frontend, text_frontend, speech_tokenizer, llm, token2wav


def get_models(use_phoneme=True, sample_rate=24000):
    """
    懒加载模型，如果采样率改变则重新加载
    """
    if MODEL_CACHE["loaded"] and MODEL_CACHE["sample_rate"] == sample_rate:
        return MODEL_CACHE["components"]
    
    # 清理旧模型
    if MODEL_CACHE["components"]:
        del MODEL_CACHE["components"]
        gc.collect()
        torch.cuda.empty_cache()

    # 加载新模型
    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
        use_phoneme=use_phoneme, 
        sample_rate=sample_rate
    )
    
    MODEL_CACHE["components"] = (frontend, text_frontend, speech_tokenizer, llm, flow)
    MODEL_CACHE["sample_rate"] = sample_rate
    MODEL_CACHE["loaded"] = True
    
    return MODEL_CACHE["components"]


# --- LLM 和 Flow 前向推理 ---

def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len


def local_llm_forward(llm, prompt_text_token, tts_text_token, prompt_speech_token, 
                      beam_size=1, sampling=25, sample_method="ras"):
    """LLM 单次前向推理"""
    prompt_text_token_len = _assert_shape_and_get_len(prompt_text_token)
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    prompt_speech_token_len = _assert_shape_and_get_len(prompt_speech_token)

    tts_speech_token = llm.inference(
        text=tts_text_token,
        text_len=tts_text_token_len,
        prompt_text=prompt_text_token,
        prompt_text_len=prompt_text_token_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        spk=None,
    )
    return tts_speech_token[0].tolist()


def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """Flow 单次前向推理"""
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel


# --- 缓存处理 ---

def get_cached_prompt(cache, synth_text_token, device=DEVICE):
    """从缓存构建 prompt tokens"""
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]

    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))

    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))

    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)

    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

    # 如果缓存太长则裁剪
    while (__len_cache_speech_token() + __len_synth_audi_token_estim > MAX_LLM_SEQ_INP_LEN):
        if len(cache_speech_token) <= 1:
            break
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)

    # 构建文本 prompt
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())

    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # 构建语音 prompt
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)

    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)

    return prompt_text_token, llm_speech_token


# --- 主生成逻辑 ---

def generate_long(frontend, text_frontend, llm, flow, text_info, cache, device,
                  embedding, seed=0, sample_method="ras", flow_prompt_token=None,
                  speech_feat=None, use_phoneme=False, skip_normalize=False):
    """长文本生成"""
    outputs = []
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]
    
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    
    # 如果 skip_normalize 为 True，说明文本已经预处理过（包括 G2P），不再分割和 normalize
    if skip_normalize:
        short_text_list = [syn_text]
    else:
        short_text_list = text_frontend.split_by_len(syn_text)

    for _, tts_text in enumerate(short_text_list):
        seed_util.set_seed(seed)
        
        # 如果 skip_normalize，直接使用传入的文本
        if skip_normalize:
            tts_text_tn = tts_text
        else:
            tts_text_tn = text_frontend.text_normalize(tts_text)
            text_tn_dict["syn_text_tn"].append(tts_text_tn)
            
            if use_phoneme:
                tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
                text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        
        tts_text_token = frontend._extract_text_token(tts_text_tn)

        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(cache, tts_text_token, device)
        else:
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor([cache_speech_token[0]], dtype=torch.int32).to(device)

        # LLM 推理
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method
        )

        output_token_list.extend(token_list_res)

        # Flow 推理
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding
        )

        # 更新缓存
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)

        outputs.append(output)
        if full_mel is not None:
            full_mels.append(full_mel)

    tts_speech = torch.concat(outputs, dim=1)
    tts_mel = torch.concat(full_mels, dim=-1) if full_mels else None

    return tts_speech, tts_mel, output_token_list, text_tn_dict


# --- Gradio 推理处理函数 ---

def run_inference(prompt_text, prompt_audio_path, input_text, seed, sample_rate, use_cache=True, use_g2p=False):
    """Gradio 主推理入口"""
    if not input_text:
        raise gr.Error("请提供要合成的文本。")
    if not prompt_audio_path:
        raise gr.Error("请上传参考音频文件。")
    if not prompt_text:
        gr.Warning("参考文本为空，合成效果可能不理想。")

    try:
        # 1. 加载模型
        frontend, text_frontend, _, llm, flow = get_models(use_phoneme=True, sample_rate=sample_rate)
        
        logging.info(f"G2P enabled: {use_g2p}")

        # 2. 文本预处理
        norm_prompt_text = text_frontend.text_normalize(prompt_text) + ' '
        norm_input_text = text_frontend.text_normalize(input_text)
        
        # 如果启用 G2P，对输入文本进行音素转换（用于多音字处理）
        if use_g2p:
            norm_input_text = text_frontend.g2p_infer(norm_input_text)
            logging.info(f"G2P processed text: {norm_input_text}")
        
        logging.info(f"Normalized Prompt: {norm_prompt_text}")
        logging.info(f"Normalized Input: {norm_input_text}")

        # 3. 特征提取
        prompt_text_token = frontend._extract_text_token(norm_prompt_text)
        prompt_speech_token = frontend._extract_speech_token([prompt_audio_path])
        speech_feat = frontend._extract_speech_feat(prompt_audio_path, sample_rate=sample_rate)
        embedding = frontend._extract_spk_embedding(prompt_audio_path)

        # 4. 准备缓存
        cache_speech_token_list = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = torch.tensor(cache_speech_token_list, dtype=torch.int32).to(DEVICE)
        
        cache = {
            'cache_text': [norm_prompt_text],
            'cache_text_token': [prompt_text_token],
            'cache_speech_token': cache_speech_token_list,
            'use_cache': use_cache
        }

        # 5. 运行生成
        tts_speech, _, _, _ = generate_long(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=flow,
            text_info=['', norm_input_text],
            cache=cache,
            embedding=embedding,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            sample_method="ras",
            seed=seed,
            device=DEVICE,
            use_phoneme=False,
            skip_normalize=True  # 文本已经在 run_inference 中预处理过
        )

        # 6. 后处理音频
        audio_data = tts_speech.squeeze().cpu().numpy()
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767.0).astype(np.int16)

        return (sample_rate, audio_int16)

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"合成失败: {str(e)}")


def clear_memory():
    """清理显存并重置模型缓存"""
    global MODEL_CACHE
    if MODEL_CACHE["components"]:
        del MODEL_CACHE["components"]
    MODEL_CACHE["components"] = None
    MODEL_CACHE["loaded"] = False
    MODEL_CACHE["sample_rate"] = None
    
    gc.collect()
    torch.cuda.empty_cache()
    return "显存已清理，模型将在下次推理时重新加载。"


def load_examples():
    """从 examples 目录加载示例数据"""
    import json
    examples_dir = os.path.join(CURRENT_DIR, "examples")
    examples = []
    
    # 定义示例文件和对应的标签，每种语言只取一个
    example_files = [
        ("example_zh.jsonl", "中文示例"),
        ("example_en.jsonl", "英文示例"),
    ]
    
    for filename, label in example_files:
        filepath = os.path.join(examples_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 1:  # 每个文件只取1个示例
                        break
                    try:
                        item = json.loads(line.strip())
                        prompt_audio_path = os.path.join(CURRENT_DIR, item["prompt_speech"])
                        if os.path.exists(prompt_audio_path):
                            examples.append([
                                item["prompt_text"],
                                prompt_audio_path,
                                item["syn_text"],
                            ])
                    except:
                        continue
    
    # 添加自定义的多音字示例（乡音无改鬓毛衰 - 衰读 cuī）
    custom_prompt_audio = os.path.join(CURRENT_DIR, "examples", "prompt", "jiayan_zh.wav")
    if os.path.exists(custom_prompt_audio):
        examples.insert(1, [
            "他当时还跟线下其他的站姐吵架，然后，打架进局子了。",
            custom_prompt_audio,
            "少小离家老大回，乡音无改鬓毛衰。儿童相见不相识，笑问客从何处来。",
        ])
    
    return examples


# --- Gradio UI 布局 ---

def create_ui():
    """创建 Gradio 界面"""
    
    # 检查默认参考音频是否存在
    default_prompt_audio = os.path.join(CURRENT_DIR, "examples", "prompt", "jiayan_zh.wav")
    if not os.path.exists(default_prompt_audio):
        default_prompt_audio = None
    
    with gr.Blocks(title="GLM-TTS 语音合成", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎵 GLM-TTS 开源语音合成演示")
        gr.Markdown("零样本文本转语音生成 - 基于 GLM-TTS 模型")
        gr.Markdown(f"**模型路径**: `{MODEL_ROOT}`")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 零样本参考设置")
                
                prompt_audio = gr.Audio(
                    label="上传参考音频 (用于克隆音色)",
                    type="filepath",
                    value=default_prompt_audio
                )
                
                prompt_text = gr.Textbox(
                    label="参考文本",
                    placeholder="输入参考音频中说话的内容...",
                    lines=2,
                    info="准确的参考文本可以提高音色相似度",
                    value="他当时还跟线下其他的站姐吵架，然后，打架进局子了。"
                )

                gr.Markdown("### 2. 输入设置")
                input_text = gr.Textbox(
                    label="要合成的文本",
                    value="我最爱吃人参果，你喜欢吃吗？", 
                    lines=5,
                    placeholder="输入想要合成的文本内容..."
                )
                
                with gr.Accordion("高级设置", open=True):
                    sample_rate = gr.Radio(
                        choices=[24000, 32000], 
                        value=24000, 
                        label="采样率 (Hz)",
                        info="32000Hz 音质更高，但需要更多计算资源"
                    )
                    seed = gr.Number(label="随机种子", value=42, precision=0)
                    use_cache = gr.Checkbox(
                        label="使用 KV Cache", 
                        value=True, 
                        info="长文本生成时更快"
                    )
                    use_g2p = gr.Checkbox(
                        label="启用 G2P (多音字处理)", 
                        value=False, 
                        info="启用后可更准确处理多音字，如'长大'vs'长度'"
                    )

                generate_btn = gr.Button("🚀 开始合成", variant="primary", size="lg")
                clear_btn = gr.Button("🧹 清理显存", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 3. 输出结果")
                output_audio = gr.Audio(label="合成结果")
                status_msg = gr.Textbox(label="系统状态", interactive=False)

        # 示例选择
        gr.Markdown("### 📋 示例选择")
        gr.Markdown("点击下方示例可快速填充参考音频、参考文本和合成文本")
        
        example_data = load_examples()
        if example_data:
            gr.Examples(
                examples=example_data,
                inputs=[prompt_text, prompt_audio, input_text],
                label="选择示例",
                examples_per_page=6,
            )

        # 事件绑定
        generate_btn.click(
            fn=run_inference,
            inputs=[prompt_text, prompt_audio, input_text, seed, sample_rate, use_cache, use_g2p],
            outputs=[output_audio]
        )

        clear_btn.click(
            fn=clear_memory,
            inputs=None,
            outputs=[status_msg]
        )

        # 使用说明
        gr.Markdown("""
        ---
        ### 使用说明
        1. **上传参考音频**: 上传一段清晰的语音作为音色参考（建议 3-10 秒）
        2. **填写参考文本**: 输入参考音频中说话的具体内容
        3. **输入合成文本**: 输入您想要合成的文本内容
        4. **点击合成**: 点击"开始合成"按钮生成语音
        
        ### 注意事项
        - 首次运行需要加载模型，可能需要等待几分钟
        - 支持中文和英文文本合成
        - 参考音频质量越高，合成效果越好
        
        ---
        ### 🔤 多音字处理说明
        
        启用 **G2P (多音字处理)** 选项后，系统会自动将多音字转换为音素标记，确保发音准确。
        
        **自定义多音字配置文件**: `configs/G2P_replace_dict.jsonl`
        
        **格式示例**:
        ```
        {"衰": "<|SH|><|UAI1|>"}           # 衰 → shuāi (一声，衰老)
        {"乡音无改鬓毛衰": "乡音无改鬓毛<|C|><|UEI1|>"}  # 古诗中"衰"读 cuī
        {"长大": "<|ZH|><|ANG3|><|D|><|A4|>"}   # 长大 → zhǎng dà
        {"长度": "<|CH|><|ANG2|><|D|><|U4|>"}   # 长度 → cháng dù
        ```
        
        **音素格式**: `<|声母|><|韵母+声调|>`
        - 声调: 1=一声, 2=二声, 3=三声, 4=四声, 5=轻声
        - 示例: `<|CH|><|ANG2|>` = cháng (二声)
        """)

    return app


# --- 主入口 ---

if __name__ == "__main__":
    # 检查模型路径
    if not os.path.exists(MODEL_ROOT):
        print(f"错误: 模型目录不存在: {MODEL_ROOT}")
        print("请确保已下载模型并放置在 checkpoints/GLM-TTS 目录下")
        exit(1)
    
    # 检查必要的模型文件
    required_paths = [
        SPEECH_TOKENIZER_PATH,
        LLM_PATH,
        FLOW_CKPT_PATH,
        FLOW_CONFIG_PATH,
        TOKENIZER_PATH,
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"错误: 模型文件不存在: {path}")
            exit(1)
    
    print(f"模型目录: {MODEL_ROOT}")
    print(f"设备: {DEVICE}")
    print("正在启动 GLM-TTS Web 服务...")
    
    app = create_ui()
    app.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False
    )
