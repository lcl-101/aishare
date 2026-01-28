# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS Gradio Web Demo
整合三种模型功能：
1. 语音设计 (Voice Design) - 通过自然语言描述生成声音
2. 语音克隆 (Voice Clone) - 使用参考音频克隆声音  
3. 预设音色 (Custom Voice) - 使用预定义的9种音色
"""

import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# ============= 模型路径配置 =============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

VOICE_DESIGN_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
CUSTOM_VOICE_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
VOICE_CLONE_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "Qwen3-TTS-12Hz-1.7B-Base")

# 设备配置
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
ATTN_IMPL = "flash_attention_2"

# ============= 全局模型变量 =============
voice_design_model = None
custom_voice_model = None
voice_clone_model = None

# ============= 语言配置 =============
SUPPORTED_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", 
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
]

LANGUAGE_DISPLAY = {
    "Auto": "自动检测",
    "Chinese": "中文",
    "English": "英语",
    "Japanese": "日语",
    "Korean": "韩语",
    "German": "德语",
    "French": "法语",
    "Russian": "俄语",
    "Portuguese": "葡萄牙语",
    "Spanish": "西班牙语",
    "Italian": "意大利语"
}

# ============= 预设音色配置 =============
CUSTOM_SPEAKERS = {
    "Vivian": {"name": "Vivian", "desc": "明亮的年轻女性声音", "lang": "Chinese"},
    "Serena": {"name": "Serena", "desc": "温暖柔和的年轻女性声音", "lang": "Chinese"},
    "Uncle_Fu": {"name": "Uncle_Fu", "desc": "成熟低沉的男性声音", "lang": "Chinese"},
    "Dylan": {"name": "Dylan", "desc": "年轻的北京男性声音", "lang": "Chinese"},
    "Eric": {"name": "Eric", "desc": "活泼的成都男性声音", "lang": "Chinese"},
    "Ryan": {"name": "Ryan", "desc": "充满活力的男性声音", "lang": "English"},
    "Aiden": {"name": "Aiden", "desc": "阳光的美式男性声音", "lang": "English"},
    "Ono_Anna": {"name": "Ono_Anna", "desc": "俏皮的日本女性声音", "lang": "Japanese"},
    "Sohee": {"name": "Sohee", "desc": "温暖的韩国女性声音", "lang": "Korean"},
}

# ============= 示例数据 =============
VOICE_DESIGN_EXAMPLES = [
    {
        "text": "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
        "language": "Chinese",
        "instruct": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。"
    },
    {
        "text": "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
        "language": "English",
        "instruct": "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    },
    {
        "text": "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?",
        "language": "English",
        "instruct": "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
    },
    {
        "text": "各位观众朋友大家好，欢迎收看今天的新闻联播。",
        "language": "Chinese",
        "instruct": "成熟稳重的男性播音员，声音洪亮有力，语速适中，字正腔圆。"
    },
]

TTS_SAMPLE_TEXTS = [
    "你好，很高兴认识你！今天的天气真的很不错。",
    "Hello, nice to meet you! The weather is really nice today.",
    "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "人工智能正在改变我们的生活方式，未来将会更加智能化。",
    "Welcome to our channel! Don't forget to like and subscribe.",
    "こんにちは、今日はいい天気ですね。",
    "안녕하세요, 만나서 반갑습니다!",
]


# ============= 工具函数 =============
def _normalize_audio(wav, eps=1e-12, clip=True):
    """规范化音频数据"""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"不支持的数据类型: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)
    
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    """将Gradio音频格式转换为(wav, sr)元组"""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    """将wav数组转换为Gradio音频格式"""
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav


# ============= 模型加载 =============
def load_models():
    """加载所有模型"""
    global voice_design_model, custom_voice_model, voice_clone_model
    
    print("\n" + "="*60)
    print("Qwen3-TTS 语音合成系统启动中...")
    print("="*60 + "\n")
    
    print("📦 [1/3] 正在加载语音设计模型 (VoiceDesign)...")
    voice_design_model = Qwen3TTSModel.from_pretrained(
        VOICE_DESIGN_MODEL_PATH,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN_IMPL,
    )
    print("✅ 语音设计模型加载完成！\n")
    
    print("📦 [2/3] 正在加载预设音色模型 (CustomVoice)...")
    custom_voice_model = Qwen3TTSModel.from_pretrained(
        CUSTOM_VOICE_MODEL_PATH,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN_IMPL,
    )
    print("✅ 预设音色模型加载完成！\n")
    
    print("📦 [3/3] 正在加载语音克隆模型 (Base/Clone)...")
    voice_clone_model = Qwen3TTSModel.from_pretrained(
        VOICE_CLONE_MODEL_PATH,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN_IMPL,
    )
    print("✅ 语音克隆模型加载完成！\n")
    
    print("="*60)
    print("🎉 所有模型加载完成，系统就绪！")
    print("="*60 + "\n")


# ============= Tab1: 语音设计功能 =============
def generate_voice_design(text: str, language: str, instruct: str):
    """语音设计生成函数"""
    try:
        if not text or not text.strip():
            return None, "❌ 错误：请输入待合成的文本内容。"
        if not instruct or not instruct.strip():
            return None, "❌ 错误：请输入声音描述（音色设计指令）。"
        
        # 获取实际语言值
        lang_key = language.split(" ")[0] if " " in language else language
        for key, display in LANGUAGE_DISPLAY.items():
            if display in language or key in language:
                lang_key = key
                break
        
        wavs, sr = voice_design_model.generate_voice_design(
            text=text.strip(),
            language=lang_key,
            instruct=instruct.strip(),
            max_new_tokens=2048,
        )
        
        return _wav_to_gradio_audio(wavs[0], sr), "✅ 语音设计生成完成！"
    
    except Exception as e:
        return None, f"❌ 生成失败：{type(e).__name__}: {e}"


def send_to_tts_tab(audio):
    """将语音设计结果发送到TTS Tab"""
    if audio is None:
        return None, "❌ 没有可发送的音频，请先生成语音。"
    return audio, "✅ 音频已发送到TTS页面！请切换到【语音合成 TTS】标签页继续操作。"


# ============= Tab2: 语音合成 TTS 功能 =============
def tts_with_custom_voice(text: str, language: str, speaker: str, instruct: str):
    """使用预设音色进行TTS"""
    try:
        if not text or not text.strip():
            return None, "❌ 错误：请输入待合成的文本内容。"
        if not speaker:
            return None, "❌ 错误：请选择说话人音色。"
        
        # 获取实际语言值
        lang_key = language.split(" ")[0] if " " in language else language
        for key, display in LANGUAGE_DISPLAY.items():
            if display in language or key in language:
                lang_key = key
                break
        
        wavs, sr = custom_voice_model.generate_custom_voice(
            text=text.strip(),
            language=lang_key,
            speaker=speaker,
            instruct=(instruct.strip() if instruct and instruct.strip() else None),
            max_new_tokens=2048,
        )
        
        return _wav_to_gradio_audio(wavs[0], sr), "✅ 语音合成完成！"
    
    except Exception as e:
        return None, f"❌ 生成失败：{type(e).__name__}: {e}"


def tts_with_voice_clone(ref_audio, ref_text: str, use_xvec: bool, text: str, language: str):
    """使用语音克隆进行TTS"""
    try:
        if not text or not text.strip():
            return None, "❌ 错误：请输入待合成的文本内容。"
        
        audio_tuple = _audio_to_tuple(ref_audio)
        if audio_tuple is None:
            return None, "❌ 错误：请上传参考音频或从语音设计接收音频。"
        
        if (not use_xvec) and (not ref_text or not ref_text.strip()):
            return None, "❌ 错误：未勾选【仅使用说话人向量】时，必须提供参考音频的文字内容。\n提示：如不想输入文字内容，请勾选【仅使用说话人向量】选项（但效果可能降低）。"
        
        # 获取实际语言值
        lang_key = language.split(" ")[0] if " " in language else language
        for key, display in LANGUAGE_DISPLAY.items():
            if display in language or key in language:
                lang_key = key
                break
        
        wavs, sr = voice_clone_model.generate_voice_clone(
            text=text.strip(),
            language=lang_key,
            ref_audio=audio_tuple,
            ref_text=(ref_text.strip() if ref_text else None),
            x_vector_only_mode=bool(use_xvec),
            max_new_tokens=2048,
        )
        
        return _wav_to_gradio_audio(wavs[0], sr), "✅ 语音克隆合成完成！"
    
    except Exception as e:
        return None, f"❌ 生成失败：{type(e).__name__}: {e}"


# ============= Gradio 界面构建 =============
# CSS 样式
CUSTOM_CSS = """
.gradio-container {max-width: none !important;}
.youtube-banner {
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.youtube-banner a {
    color: white !important;
    text-decoration: none;
    font-weight: bold;
    font-size: 1.1em;
}
.youtube-banner a:hover {
    text-decoration: underline;
}
"""

def build_demo():
    """构建Gradio界面"""
    
    # 语言选择列表（带中文说明）
    lang_choices = [f"{lang} ({LANGUAGE_DISPLAY[lang]})" for lang in SUPPORTED_LANGUAGES]
    
    # 预设音色列表
    speaker_choices = [f"{info['name']} - {info['desc']}" for name, info in CUSTOM_SPEAKERS.items()]
    
    with gr.Blocks(title="Qwen3-TTS 语音合成系统") as demo:
        
        # YouTube 频道横幅
        gr.HTML("""
        <style>
        .gradio-container {max-width: none !important;}
        .youtube-banner {
            background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .youtube-banner a {
            color: white !important;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.1em;
        }
        .youtube-banner a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="youtube-banner">
            <h2 style="margin: 0 0 10px 0;">🎬 AI 技术分享频道</h2>
            <p style="margin: 0; font-size: 0.95em;">
                欢迎订阅我的 YouTube 频道，获取更多 AI 技术教程和项目演示！
                <br><br>
                <a href="https://www.youtube.com/@rongyi-ai" target="_blank">
                    📺 https://www.youtube.com/@rongyi-ai
                </a>
            </p>
        </div>
        """)
        
        gr.Markdown("""
# 🎙️ Qwen3-TTS 语音合成系统

**Qwen3-TTS** 是阿里云通义团队推出的强大语音生成模型系列：
- 🎨 **语音设计 (Voice Design)**：通过自然语言描述，凭空生成符合描述的声音
- 🔊 **语音克隆 (Voice Clone)**：仅需3秒参考音频，即可模仿说话人声音 (Zero-shot)
- 🎭 **预设音色 (Custom Voice)**：9种精选音色，支持多语言和方言
- ⚡ **超低延迟**：首包延迟仅97ms，适合实时语音交互
- 🌍 **多语言支持**：支持中文、英语、日语、韩语等10种语言
        """)
        
        # 用于跨Tab传递音频的State
        shared_audio_state = gr.State(None)
        
        with gr.Tabs():
            # =============== Tab 1: 语音设计 ===============
            with gr.Tab("🎨 语音设计", id="voice_design"):
                gr.Markdown("""
### 语音设计 (Voice Design)
通过自然语言描述您想要的声音特征，AI模型会凭空生成符合描述的声音。  
例如：*"一个年轻女性的声音，语气兴奋，语速很快"*、*"Male, 17 years old, tenor range, gaining confidence"*

生成的声音可以发送到【语音合成 TTS】页面，用作参考音频进行克隆。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**📝 快速示例**")
                        
                        example_btns = []
                        for i, ex in enumerate(VOICE_DESIGN_EXAMPLES):
                            btn_text = f"示例 {i+1}: {ex['text'][:25]}..."
                            example_btns.append(gr.Button(btn_text, size="sm"))
                        
                        design_text = gr.Textbox(
                            label="待合成文本",
                            lines=3,
                            placeholder="请输入您想要合成的文本内容...",
                            value=VOICE_DESIGN_EXAMPLES[0]["text"]
                        )
                        
                        design_lang = gr.Dropdown(
                            label="语言",
                            choices=lang_choices,
                            value=f"Chinese ({LANGUAGE_DISPLAY['Chinese']})",
                        )
                        
                        design_instruct = gr.Textbox(
                            label="声音描述（Voice Design Instruction）- 保持原始提示词语言",
                            lines=4,
                            placeholder="请用自然语言描述您想要的声音特征...",
                            value=VOICE_DESIGN_EXAMPLES[0]["instruct"],
                        )
                        
                        with gr.Row():
                            design_gen_btn = gr.Button("🎵 生成语音", variant="primary", scale=2)
                            design_send_btn = gr.Button("📤 发送到TTS", variant="secondary", scale=1)
                    
                    with gr.Column(scale=1):
                        design_audio_out = gr.Audio(
                            label="生成的音频",
                            type="numpy",
                        )
                        design_status = gr.Textbox(
                            label="状态信息",
                            lines=3,
                            value="💡 提示：点击左侧示例按钮快速体验",
                            interactive=False
                        )
                
                # 示例按钮事件
                for i, btn in enumerate(example_btns):
                    ex = VOICE_DESIGN_EXAMPLES[i]
                    lang_val = f"{ex['language']} ({LANGUAGE_DISPLAY.get(ex['language'], ex['language'])})"
                    btn.click(
                        fn=lambda t=ex['text'], l=lang_val, inst=ex['instruct']: (t, l, inst),
                        outputs=[design_text, design_lang, design_instruct]
                    )
                
                # 生成按钮
                design_gen_btn.click(
                    fn=generate_voice_design,
                    inputs=[design_text, design_lang, design_instruct],
                    outputs=[design_audio_out, design_status]
                )
                
                # 发送按钮
                design_send_btn.click(
                    fn=send_to_tts_tab,
                    inputs=[design_audio_out],
                    outputs=[shared_audio_state, design_status]
                )
            
            # =============== Tab 2: 语音合成 TTS ===============
            with gr.Tab("🔊 语音合成 TTS", id="tts"):
                gr.Markdown("""
### 语音合成 (Text-to-Speech)
使用以下三种方式进行语音合成：
1. **预设音色**：选择9种精选音色之一（支持多语言和方言）
2. **语音克隆**：上传或录制参考音频（建议3秒以上）
3. **使用设计的声音**：从【语音设计】页面发送过来的音频
                """)
                
                with gr.Tabs():
                    # 子Tab 1: 预设音色
                    with gr.Tab("🎭 预设音色"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**音色选择与设置**")
                                
                                custom_speaker = gr.Dropdown(
                                    label="选择说话人（9种精选音色）",
                                    choices=speaker_choices,
                                    value=speaker_choices[0],
                                )
                                
                                custom_lang = gr.Dropdown(
                                    label="语言",
                                    choices=lang_choices,
                                    value=f"Auto ({LANGUAGE_DISPLAY['Auto']})",
                                )
                                
                                custom_instruct = gr.Textbox(
                                    label="控制指令（可选，用于控制情绪、语气等）",
                                    lines=2,
                                    placeholder="例如：用特别愤怒的语气说 / Very happy",
                                )
                                
                                # 示例文本选择
                                sample_dropdown = gr.Dropdown(
                                    label="选择示例文本",
                                    choices=[f"示例 {i+1}: {t[:30]}..." if len(t) > 30 else f"示例 {i+1}: {t}" 
                                             for i, t in enumerate(TTS_SAMPLE_TEXTS)],
                                )
                                
                                custom_text = gr.Textbox(
                                    label="待合成文本",
                                    lines=4,
                                    placeholder="请输入您想要合成的文本...",
                                )
                                
                                custom_gen_btn = gr.Button("🎵 生成语音", variant="primary")
                            
                            with gr.Column(scale=1):
                                custom_audio_out = gr.Audio(label="生成的音频", type="numpy")
                                custom_status = gr.Textbox(
                                    label="状态信息",
                                    lines=3,
                                    value="💡 提示：选择一个说话人和示例文本开始",
                                    interactive=False
                                )
                        
                        # 示例文本选择事件
                        sample_dropdown.change(
                            fn=lambda x: TTS_SAMPLE_TEXTS[int(x.split(":")[0].replace("示例 ", "")) - 1] if x else "",
                            inputs=[sample_dropdown],
                            outputs=[custom_text]
                        )
                        
                        # 生成按钮
                        custom_gen_btn.click(
                            fn=lambda text, lang, spk, inst: tts_with_custom_voice(
                                text, lang, spk.split(" - ")[0], inst
                            ),
                            inputs=[custom_text, custom_lang, custom_speaker, custom_instruct],
                            outputs=[custom_audio_out, custom_status]
                        )
                    
                    # 子Tab 2: 语音克隆
                    with gr.Tab("🎤 语音克隆"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**参考音频设置**（上传或录制参考音频，建议3秒以上）")
                                
                                clone_ref_audio = gr.Audio(
                                    label="参考音频",
                                    type="numpy",
                                    sources=["upload", "microphone"],
                                )
                                
                                receive_btn = gr.Button("📥 使用语音设计的结果", variant="secondary")
                                
                                clone_ref_text = gr.Textbox(
                                    label="参考音频文字内容（推荐填写以获得更好的克隆效果）",
                                    lines=2,
                                    placeholder="参考音频中说话人所说的文字内容...",
                                    value="从古希腊到启蒙运动，西方思想一直在追寻某种终极目标，无论是上帝，理性还是人类的完美。",
                                )
                                
                                clone_xvec = gr.Checkbox(
                                    label="仅使用说话人向量（不需要填写参考文字，但效果可能降低）",
                                    value=False,
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("**合成设置**")
                                
                                clone_sample_dropdown = gr.Dropdown(
                                    label="选择示例文本",
                                    choices=[f"示例 {i+1}: {t[:30]}..." if len(t) > 30 else f"示例 {i+1}: {t}" 
                                             for i, t in enumerate(TTS_SAMPLE_TEXTS)],
                                )
                                
                                clone_text = gr.Textbox(
                                    label="待合成文本",
                                    lines=4,
                                    placeholder="请输入您想要用克隆声音说出的文本...",
                                )
                                
                                clone_lang = gr.Dropdown(
                                    label="语言",
                                    choices=lang_choices,
                                    value=f"Auto ({LANGUAGE_DISPLAY['Auto']})",
                                )
                                
                                clone_gen_btn = gr.Button("🎵 生成语音", variant="primary")
                                
                                clone_audio_out = gr.Audio(label="生成的音频", type="numpy")
                                clone_status = gr.Textbox(
                                    label="状态信息",
                                    lines=3,
                                    value="💡 提示：上传参考音频或使用语音设计的结果",
                                    interactive=False
                                )
                        
                        # 接收语音设计结果
                        receive_btn.click(
                            fn=lambda x: (x, "✅ 已加载语音设计的结果！") if x is not None else (
                                None, "❌ 没有可用的语音设计结果\n请先在【语音设计】页面生成并发送"),
                            inputs=[shared_audio_state],
                            outputs=[clone_ref_audio, clone_status]
                        )
                        
                        # 示例文本选择
                        clone_sample_dropdown.change(
                            fn=lambda x: TTS_SAMPLE_TEXTS[int(x.split(":")[0].replace("示例 ", "")) - 1] if x else "",
                            inputs=[clone_sample_dropdown],
                            outputs=[clone_text]
                        )
                        
                        # 生成按钮
                        clone_gen_btn.click(
                            fn=tts_with_voice_clone,
                            inputs=[clone_ref_audio, clone_ref_text, clone_xvec, clone_text, clone_lang],
                            outputs=[clone_audio_out, clone_status]
                        )
        
        # 免责声明
        gr.Markdown("""
---
### ⚠️ 免责声明

本音频由人工智能模型自动生成/合成，仅用于技术演示和学习目的，可能存在不准确或不当之处。
其内容不代表开发者立场，亦不构成任何专业建议。用户应自行评估并承担使用、传播或依赖该音频所产生的一切风险与责任。
**严禁利用本服务生成违法、有害、诽谤、欺诈、深度伪造、侵犯隐私/肖像/著作权等内容。**
        """)
    
    return demo


# ============= 主函数 =============
def main():
    """主函数"""
    # 加载模型
    load_models()
    
    # 构建并启动Demo
    print("🚀 正在启动 Gradio 服务...")
    demo = build_demo()
    demo.queue(default_concurrency_limit=16).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
