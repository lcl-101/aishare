#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CosyVoice3 Gradio Web Application
基于 Fun-CosyVoice3-0.5B 的语音合成 Web 应用

功能:
1. Zero-Shot 语音克隆 (3秒极速复刻)
2. 跨语种/细粒度控制
3. 指令控制语音合成

使用方法:
    python app.py
    python app.py --port 7860 --share
"""

import os
import sys
import argparse
import random

import gradio as gr
import numpy as np
import torch
import torchaudio

# 添加 Matcha-TTS 到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

# 创建 pretrained_models 软链接指向 checkpoints（ttsfrd 需要）
pretrained_models_path = os.path.join(ROOT_DIR, 'pretrained_models')
checkpoints_path = os.path.join(ROOT_DIR, 'checkpoints')
if not os.path.exists(pretrained_models_path) and os.path.exists(checkpoints_path):
    os.symlink(checkpoints_path, pretrained_models_path)
    print(f"已创建软链接: pretrained_models -> checkpoints")

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# 默认模型路径
DEFAULT_MODEL_DIR = "checkpoints/Fun-CosyVoice3-0.5B-2512"

# 全局变量
cosyvoice = None
sample_rate = 22050


def generate_seed():
    """生成随机种子"""
    return random.randint(1, 100000000)


def load_model(model_dir: str):
    """加载模型"""
    global cosyvoice, sample_rate
    
    if not os.path.exists(model_dir):
        return f"❌ 模型路径不存在: {model_dir}", gr.update()
    
    try:
        logging.info(f"正在加载模型: {model_dir}")
        cosyvoice = AutoModel(model_dir=model_dir)
        sample_rate = cosyvoice.sample_rate
        
        spks = cosyvoice.list_available_spks()
        spk_choices = spks if spks else ["无预训练音色"]
        
        return f"✅ 模型加载成功！\n采样率: {sample_rate}Hz\n预训练音色: {', '.join(spks) if spks else '无'}", \
               gr.update(choices=spk_choices, value=spk_choices[0] if spk_choices else None)
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return f"❌ 模型加载失败: {str(e)}", gr.update()


def inference_zero_shot(tts_text, prompt_text, prompt_audio, seed, speed):
    """
    Zero-Shot 语音克隆
    CosyVoice3 格式: prompt_text 需要加上 system prompt
    """
    if cosyvoice is None:
        gr.Warning("请先加载模型！")
        return None
    
    if not tts_text.strip():
        gr.Warning("请输入要合成的文本！")
        return None
    
    if prompt_audio is None:
        gr.Warning("请上传参考音频！")
        return None
    
    if not prompt_text.strip():
        gr.Warning("请输入参考音频对应的文本！")
        return None
    
    # CosyVoice3 需要添加 system prompt
    full_prompt_text = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
    
    logging.info(f"Zero-Shot 推理: tts_text={tts_text[:30]}...")
    set_all_random_seed(seed)
    
    try:
        # 收集所有输出
        audio_segments = []
        for result in cosyvoice.inference_zero_shot(
            tts_text, full_prompt_text, prompt_audio, stream=False
        ):
            audio_segments.append(result['tts_speech'])
        
        if audio_segments:
            # 合并音频
            full_audio = torch.cat(audio_segments, dim=1)
            return (sample_rate, full_audio.numpy().flatten())
        return None
    except Exception as e:
        logging.error(f"推理失败: {e}")
        gr.Error(f"推理失败: {str(e)}")
        return None


def inference_cross_lingual(tts_text, prompt_audio, seed, speed):
    """
    跨语种/细粒度控制
    支持控制标签: [breath], [laughter] 等
    """
    if cosyvoice is None:
        gr.Warning("请先加载模型！")
        return None
    
    if not tts_text.strip():
        gr.Warning("请输入要合成的文本！")
        return None
    
    if prompt_audio is None:
        gr.Warning("请上传参考音频！")
        return None
    
    # CosyVoice3 格式
    if not tts_text.startswith("You are"):
        tts_text = f"You are a helpful assistant.<|endofprompt|>{tts_text}"
    
    logging.info(f"跨语种推理: tts_text={tts_text[:50]}...")
    set_all_random_seed(seed)
    
    try:
        audio_segments = []
        for result in cosyvoice.inference_cross_lingual(
            tts_text, prompt_audio, stream=False
        ):
            audio_segments.append(result['tts_speech'])
        
        if audio_segments:
            full_audio = torch.cat(audio_segments, dim=1)
            return (sample_rate, full_audio.numpy().flatten())
        return None
    except Exception as e:
        logging.error(f"推理失败: {e}")
        gr.Error(f"推理失败: {str(e)}")
        return None


def inference_instruct(tts_text, instruct_text, prompt_audio, seed, speed):
    """
    指令控制合成
    例如: 用四川话说、用广东话表达、用快速语速说等
    """
    if cosyvoice is None:
        gr.Warning("请先加载模型！")
        return None
    
    if not tts_text.strip():
        gr.Warning("请输入要合成的文本！")
        return None
    
    if not instruct_text.strip():
        gr.Warning("请输入指令文本！")
        return None
    
    if prompt_audio is None:
        gr.Warning("请上传参考音频！")
        return None
    
    # 确保指令文本格式正确
    if not instruct_text.startswith("You are"):
        instruct_text = f"You are a helpful assistant. {instruct_text}"
    if not instruct_text.endswith("<|endofprompt|>"):
        instruct_text = f"{instruct_text}<|endofprompt|>"
    
    logging.info(f"指令控制推理: tts_text={tts_text[:30]}..., instruct={instruct_text[:50]}...")
    set_all_random_seed(seed)
    
    try:
        audio_segments = []
        for result in cosyvoice.inference_instruct2(
            tts_text, instruct_text, prompt_audio, stream=False
        ):
            audio_segments.append(result['tts_speech'])
        
        if audio_segments:
            full_audio = torch.cat(audio_segments, dim=1)
            return (sample_rate, full_audio.numpy().flatten())
        return None
    except Exception as e:
        logging.error(f"推理失败: {e}")
        gr.Error(f"推理失败: {str(e)}")
        return None


def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="CosyVoice3 语音合成", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎤 CosyVoice3 语音合成系统
            
            基于阿里通义实验室 **Fun-CosyVoice3-0.5B** 模型
            
            📖 [GitHub](https://github.com/FunAudioLLM/CosyVoice) | 
            [CosyVoice3 介绍](https://funaudiollm.github.io/cosyvoice3/)
            """     
        )
        
        # 功能选项卡
        with gr.Tabs():
            # Tab 1: Zero-Shot
            with gr.TabItem("🎭 3秒极速复刻"):
                gr.Markdown(
                    """
                    上传 **3-10秒** 参考音频，输入对应文本，即可克隆音色合成新语音。
                    """
                )
                with gr.Row():
                    with gr.Column():
                        zs_prompt_audio = gr.Audio(
                            label="📁 参考音频（3-30秒）",
                            sources=["upload", "microphone"],
                            type="filepath"
                        )
                        zs_prompt_text = gr.Textbox(
                            label="参考音频文本",
                            placeholder="请输入参考音频中说的内容...",
                            lines=2,
                            value="希望你以后能够做的比我还好呦。"
                        )
                    with gr.Column():
                        zs_tts_text = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要合成的文本...",
                            lines=4,
                            value="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
                        )
                        with gr.Row():
                            zs_seed = gr.Number(label="随机种子", value=42, precision=0)
                            zs_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
                
                zs_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
                zs_output = gr.Audio(label="合成结果", type="numpy")
                
                # 示例
                gr.Examples(
                    examples=[
                        ["八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。", 
                         "希望你以后能够做的比我还好呦。",
                         "./asset/zero_shot_prompt.wav"],
                    ],
                    inputs=[zs_tts_text, zs_prompt_text, zs_prompt_audio],
                    label="示例"
                )
                
                zs_btn.click(
                    fn=inference_zero_shot,
                    inputs=[zs_tts_text, zs_prompt_text, zs_prompt_audio, zs_seed, zs_speed],
                    outputs=[zs_output]
                )
            
            # Tab 2: 跨语种复刻
            with gr.TabItem("🌍 跨语种复刻"):
                gr.Markdown(
                    """
                    用不同语言合成语音，保持参考音频的音色特征。
                    
                    **CosyVoice3 支持多语言混合**，直接输入目标语言文本即可，无需语言标签。
                    """
                )
                with gr.Row():
                    with gr.Column():
                        xl_prompt_audio = gr.Audio(
                            label="📁 参考音频（提供音色）",
                            sources=["upload", "microphone"],
                            type="filepath",
                            value="./asset/cross_lingual_prompt.wav"
                        )
                    with gr.Column():
                        xl_tts_text = gr.Textbox(
                            label="合成文本（直接输入目标语言）",
                            placeholder="直接输入英文、日文、中文等文本...",
                            lines=4,
                            value="And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that's coming into the family."
                        )
                        with gr.Row():
                            xl_seed = gr.Number(label="随机种子", value=42, precision=0)
                            xl_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
                
                xl_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
                xl_output = gr.Audio(label="合成结果", type="numpy")
                
                # 跨语种示例
                gr.Examples(
                    examples=[
                        ["Hello, I am a text to speech model. How can I help you today?", "./asset/cross_lingual_prompt.wav"],
                        ["こんにちは、今日はいい天気ですね。お元気ですか？", "./asset/cross_lingual_prompt.wav"],
                        ["This is a mixed language test. 这是中英混合测试。很高兴认识你。", "./asset/cross_lingual_prompt.wav"],
                    ],
                    inputs=[xl_tts_text, xl_prompt_audio],
                    label="📌 点击示例快速填充（英文/日文/中英混合）"
                )
                
                xl_btn.click(
                    fn=inference_cross_lingual,
                    inputs=[xl_tts_text, xl_prompt_audio, xl_seed, xl_speed],
                    outputs=[xl_output]
                )
            
            # Tab 3: 细粒度控制
            with gr.TabItem("🎛️ 细粒度控制"):
                gr.Markdown(
                    """
                    使用控制标签精细控制语音合成效果。
                    
                    **支持的标签**: `[breath]` 呼吸声, `[laughter]` 笑声 等
                    """
                )
                with gr.Row():
                    with gr.Column():
                        cl_prompt_audio = gr.Audio(
                            label="📁 参考音频",
                            sources=["upload", "microphone"],
                            type="filepath",
                            value="./asset/zero_shot_prompt.wav"
                        )
                    with gr.Column():
                        cl_tts_text = gr.Textbox(
                            label="合成文本（可包含控制标签）",
                            placeholder="例如: [breath]因为他们那一辈人[breath]在乡里面住...",
                            lines=4,
                            value="[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]"
                        )
                        with gr.Row():
                            cl_seed = gr.Number(label="随机种子", value=42, precision=0)
                            cl_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
                
                cl_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
                cl_output = gr.Audio(label="合成结果", type="numpy")
                
                # 细粒度控制示例
                gr.Examples(
                    examples=[
                        ["[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]", "./asset/zero_shot_prompt.wav"],
                        ["在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。", "./asset/zero_shot_prompt.wav"],
                    ],
                    inputs=[cl_tts_text, cl_prompt_audio],
                    label="📌 点击示例快速填充"
                )
                
                cl_btn.click(
                    fn=inference_cross_lingual,
                    inputs=[cl_tts_text, cl_prompt_audio, cl_seed, cl_speed],
                    outputs=[cl_output]
                )
            
            # Tab 4: 指令控制
            with gr.TabItem("📝 指令控制"):
                gr.Markdown(
                    """
                    使用自然语言指令控制语音风格、方言等。
                    """
                )
                with gr.Row():
                    with gr.Column():
                        inst_prompt_audio = gr.Audio(
                            label="📁 参考音频",
                            sources=["upload", "microphone"],
                            type="filepath",
                            value="./asset/zero_shot_prompt.wav"
                        )
                        inst_instruct = gr.Textbox(
                            label="指令",
                            placeholder="例如: 请用广东话表达",
                            lines=2,
                            value="请用广东话表达"
                        )
                    with gr.Column():
                        inst_tts_text = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要合成的文本...",
                            lines=4,
                            value="好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。"
                        )
                        with gr.Row():
                            inst_seed = gr.Number(label="随机种子", value=42, precision=0)
                            inst_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
                
                inst_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
                inst_output = gr.Audio(label="合成结果", type="numpy")
                
                # 指令示例
                gr.Examples(
                    examples=[
                        ["好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。", "请用广东话表达", "./asset/zero_shot_prompt.wav"],
                        ["收到好友从远方寄来的生日礼物，那份意外的惊喜让我心中充满了快乐。", "请用四川话说这句话", "./asset/zero_shot_prompt.wav"],
                        ["收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "请用尽可能快的语速说", "./asset/zero_shot_prompt.wav"],
                        ["收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。", "请用温柔的语气说", "./asset/zero_shot_prompt.wav"],
                    ],
                    inputs=[inst_tts_text, inst_instruct, inst_prompt_audio],
                    label="📌 点击示例快速填充"
                )
                
                inst_btn.click(
                    fn=inference_instruct,
                    inputs=[inst_tts_text, inst_instruct, inst_prompt_audio, inst_seed, inst_speed],
                    outputs=[inst_output]
                )
        
        gr.Markdown(
            """
            ---
            ### 📁 示例音频文件
            - `./asset/zero_shot_prompt.wav` - 参考音频示例
            - `./asset/cross_lingual_prompt.wav` - 跨语种参考音频
            """
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 Web UI")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="模型路径")
    args = parser.parse_args()
    
    # 启动时直接加载模型
    print(f"正在加载模型: {args.model_dir}")
    status, _ = load_model(args.model_dir)
    print(status)
    
    if cosyvoice is None:
        print("模型加载失败，请检查模型路径！")
        return
    
    # 启动 UI
    demo = create_ui()
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False
    )


if __name__ == "__main__":
    main()
