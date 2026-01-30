# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Gradio Web 应用程序 - Qwen3 ASR 音频转写服务
"""

import base64
import io
import os
import urllib.request
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel


# 模型路径配置
ASR_MODEL_PATH = "checkpoints/Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = "checkpoints/Qwen3-ForcedAligner-0.6B"

# 示例音频 URL
EXAMPLE_URLS = {
    "中文示例": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    "英文示例": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
}

# 全局变量存储模型实例
asr_model = None
example_audios = {}


def download_example_audio(url: str, timeout: int = 30) -> Optional[bytes]:
    """下载示例音频文件"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f"下载音频失败: {url}, 错误: {e}")
        return None


def init_examples():
    """初始化并下载示例音频"""
    global example_audios
    print("正在下载示例音频文件...")
    for name, url in EXAMPLE_URLS.items():
        audio_bytes = download_example_audio(url)
        if audio_bytes:
            # 保存到临时文件
            temp_path = f"/tmp/{name}.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            example_audios[name] = temp_path
            print(f"✓ {name} 下载完成")
        else:
            print(f"✗ {name} 下载失败")


def init_model():
    """初始化 ASR 模型"""
    global asr_model
    print("正在加载 Qwen3 ASR 模型...")
    asr_model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        forced_aligner=FORCED_ALIGNER_PATH,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ),
        max_inference_batch_size=32,
        max_new_tokens=2048,
    )
    print("✓ 模型加载完成")


def format_timestamps(time_stamps):
    """格式化时间戳信息"""
    if not time_stamps or len(time_stamps) == 0:
        return ""
    
    result = []
    result.append("\n## 时间戳详情：\n")
    for i, ts in enumerate(time_stamps):
        result.append(f"{i+1}. [{ts.start_time:.2f}s - {ts.end_time:.2f}s] {ts.text}")
    return "\n".join(result)


def transcribe_audio(
    audio_input,
    language: str,
    context: str,
    return_timestamps: bool
) -> Tuple[str, str]:
    """
    转写音频文件
    
    Args:
        audio_input: 音频文件路径（来自 Gradio 音频组件）
        language: 语言选择
        context: 上下文提示词
        return_timestamps: 是否返回时间戳
    
    Returns:
        (转写文本, 时间戳信息)
    """
    if asr_model is None:
        return "错误：模型未加载", ""
    
    if audio_input is None:
        return "请上传或选择音频文件", ""
    
    try:
        # 处理 Gradio 音频输入
        if isinstance(audio_input, str):
            # 文件路径
            audio_path = audio_input
        elif isinstance(audio_input, tuple):
            # (采样率, 音频数据) 元组
            sr, audio_data = audio_input
            audio_path = (audio_data.astype(np.float32) / 32768.0, sr)
        else:
            return "不支持的音频格式", ""
        
        # 设置语言参数
        lang_param = None if language == "自动检测" else language
        
        # 执行转写
        results = asr_model.transcribe(
            audio=audio_path,
            language=lang_param,
            context=context if context.strip() else "",
            return_time_stamps=return_timestamps,
        )
        
        if not results or len(results) == 0:
            return "转写失败：未返回结果", ""
        
        result = results[0]
        
        # 构建输出文本
        output_text = f"**检测语言：** {result.language}\n\n**转写文本：**\n{result.text}"
        
        # 构建时间戳信息
        timestamp_text = ""
        if return_timestamps and result.time_stamps:
            timestamp_text = format_timestamps(result.time_stamps)
        
        return output_text, timestamp_text
        
    except Exception as e:
        import traceback
        error_msg = f"转写出错：{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, ""


def load_example_audio(example_name: str):
    """加载示例音频"""
    if example_name in example_audios:
        return example_audios[example_name]
    return None


def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="Qwen3 ASR 音频转写服务", theme=gr.themes.Soft()) as demo:
        # YouTube 频道信息
        gr.Markdown(
            """
            # 🎙️ Qwen3 ASR 音频转写服务
            
            ### 📺 [AI 技术分享频道](https://www.youtube.com/@rongyi-ai) - 欢迎订阅！
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 音频输入")
                
                # 示例音频选择
                example_dropdown = gr.Dropdown(
                    choices=list(example_audios.keys()),
                    label="选择示例音频",
                    value=None,
                )
                
                load_example_btn = gr.Button("加载示例", variant="secondary")
                
                gr.Markdown("**或上传音频文件：**")
                
                # 音频输入
                audio_input = gr.Audio(
                    label="上传音频文件",
                    type="filepath",
                )
                
                gr.Markdown("## ⚙️ 转写设置")
                
                # 语言选择
                language_select = gr.Radio(
                    choices=["自动检测", "Chinese", "English"],
                    value="自动检测",
                    label="语言设置",
                )
                
                # 上下文提示词
                context_input = gr.Textbox(
                    label="上下文提示词（可选）",
                    placeholder="交易 停滞",
                    lines=2,
                )
                
                # 时间戳选项
                timestamp_checkbox = gr.Checkbox(
                    label="生成时间戳信息",
                    value=False,
                )
                
                # 转写按钮
                transcribe_btn = gr.Button("🎯 开始转写", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("## 📝 转写结果")
                
                # 转写文本输出
                output_text = gr.Markdown(
                    label="转写文本",
                    value="等待转写...",
                )
                
                # 时间戳输出
                timestamp_output = gr.Markdown(
                    label="时间戳信息",
                    value="",
                )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown(
                """
                ### 功能说明
                
                1. **选择音频**：可以从示例音频中选择，或上传自己的音频文件
                2. **语言设置**：
                   - 自动检测：让模型自动识别语言
                   - Chinese：强制使用中文识别
                   - English：强制使用英文识别
                3. **上下文提示词**：提供相关词汇可以提高识别准确率（如：交易 停滞）
                4. **时间戳信息**：勾选后会生成每个词的开始和结束时间
                
                ### 支持的音频格式
                
                - WAV, MP3, FLAC, OGG 等常见格式
                - 建议使用 16kHz 采样率的音频以获得最佳效果
                """
            )
        
        # 事件绑定
        load_example_btn.click(
            fn=load_example_audio,
            inputs=[example_dropdown],
            outputs=[audio_input],
        )
        
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, language_select, context_input, timestamp_checkbox],
            outputs=[output_text, timestamp_output],
        )
    
    return demo


def main():
    """主函数"""
    print("=" * 60)
    print("Qwen3 ASR Web 服务启动中...")
    print("=" * 60)
    
    # 初始化示例音频
    init_examples()
    
    # 初始化模型
    init_model()
    
    # 创建并启动 UI
    demo = create_ui()
    
    print("\n" + "=" * 60)
    print("✓ 服务启动成功！")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
