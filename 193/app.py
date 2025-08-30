#!/usr/bin/env python3
"""Simple Gradio WebUI for OLMoASR local inference.

Features:
 - Dropdown to select model (tiny/base/small/medium/large/large-v2) using local checkpoint files you renamed.
 - Button to (re)load model.
 - Audio upload / microphone recording.
 - Transcription with optional language selection & word timestamps toggle.
 - Shows raw text plus JSON metadata.

Run:
    python app.py
Then open the printed local URL.
"""
from __future__ import annotations
import gradio as gr
import olmoasr
import torch
import time
import json
import tempfile
from pathlib import Path
from typing import Optional

# Map model choice -> checkpoint filename expected in your local checkpoints directory
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large", "large-v2"]
DEFAULT_DIR = Path("checkpoints").resolve()

_loaded_model = None
_loaded_name = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _checkpoint_path_for(name: str) -> Path:
    # load_model(name) would try to download if file absent; we want local file directly
    # We assume you already created these via rename script: OLMoASR-{name}.pt (except large-v2)
    # large-v2 becomes OLMoASR-large-v2.pt
    return DEFAULT_DIR / f"OLMoASR-{name}.pt"


def load_selected_model(name: str, inference: bool = True) -> str:
    global _loaded_model, _loaded_name
    ckpt = _checkpoint_path_for(name)
    if not ckpt.exists():
        return f"找不到本地模型文件: {ckpt}. 请确认已重命名。"
    if _loaded_model is not None and _loaded_name == name:
        return f"模型已加载: {name}"
    t0 = time.time()
    _loaded_model = olmoasr.load_model(str(ckpt), device=_device, inference=inference)
    _loaded_name = name
    dt = time.time() - t0
    return f"模型加载完成: {name} (用时 {dt:.2f}s, 设备: {_device})"


def _format_timestamp(ts: float) -> str:
    ms = int(round(ts * 1000))
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_srt(segments):
    lines = []
    for idx, seg in enumerate(segments, 1):
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"


def transcribe_audio(audio_path: str) -> tuple[str, str, str, str]:
    """执行转写，返回: 纯文本, 字幕文本(SRT格式), SRT文件路径, JSON字符串"""
    if _loaded_model is None:
        return "请先加载模型", "", "", "{}"
    if audio_path is None:
        return "未提供音频", "", "", "{}"

    # 固定使用 deterministic 设置
    result = _loaded_model.transcribe(
        audio_path,
        temperature=(0.0,),
        language=None,  # 自动检测
        word_timestamps=False,
    )
    text = result.get("text", "")
    segments = result.get("segments", [])
    srt_text = _segments_to_srt(segments) if segments else ""

    # 写临时 SRT 文件供下载
    srt_path = ""
    if srt_text:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".srt")
        tmp.write(srt_text.encode("utf-8"))
        tmp.flush()
        tmp.close()
        srt_path = tmp.name

    meta = json.dumps(result, ensure_ascii=False, indent=2)
    return text, srt_text, srt_path, meta

with gr.Blocks(title="OLMoASR WebUI") as demo:
    gr.Markdown("# OLMoASR 本地转写 WebUI")
    gr.Markdown("选择模型 -> 加载 -> 上传或录音 -> 转写 (默认温度 0，自动语言)")

    with gr.Row():
        model_name = gr.Dropdown(choices=MODEL_CHOICES, value="medium", label="模型选择")
        load_btn = gr.Button("加载模型")
        status = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="音频")
    transcribe_btn = gr.Button("开始转写", variant="primary")

    output_text = gr.Textbox(label="转写文本", lines=6)
    subtitles_box = gr.Textbox(label="字幕 (SRT)", lines=12)
    srt_download = gr.File(label="下载 SRT 文件")
    output_json = gr.Code(label="详细结果 JSON")

    load_btn.click(fn=load_selected_model, inputs=[model_name], outputs=status)
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[output_text, subtitles_box, srt_download, output_json],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
