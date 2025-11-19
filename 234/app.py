"""Unified multi-tab demo app for Ovis 2.5 models.

This script exposes multiple inference scenarios (single image, multi-image, video,
text-only, and reflective reasoning) from a single Gradio UI. Each request starts a
fresh conversation with the underlying Ovis model.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Optional, Sequence, Tuple

import gradio as gr
import PIL.Image
import torch
import numpy as np

try:
    from moviepy.editor import VideoFileClip  # type: ignore

    _HAS_MOVIEPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_MOVIEPY = False

from ovis.model.modeling_ovis import Ovis


MODEL: Optional[Ovis] = None
MODEL_PATH: Optional[str] = None
DEVICE: Optional[str] = None


def _ensure_model(model_path: str, device: str) -> Ovis:
    """Load the model once and reuse it across tabs."""
    global MODEL, MODEL_PATH, DEVICE
    if MODEL is None or model_path != MODEL_PATH or device != DEVICE:
        print(f"Loading model from {model_path} onto {device}…")
        MODEL = Ovis.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        ).eval()
        MODEL_PATH = model_path
        DEVICE = device
        print("Model loaded.")
    return MODEL


def _files_to_images(file_paths: Optional[Sequence[str]]) -> Optional[List[PIL.Image.Image]]:
    if not file_paths:
        return None
    images: List[PIL.Image.Image] = []
    for path in file_paths:
        if not path:
            continue
        images.append(PIL.Image.open(path).convert("RGB"))
    return images or None


def _extract_video_segments(
    video_path: Optional[str],
    *,
    n_frames: int,
    segment_duration: int,
    max_segments: int,
) -> Tuple[List[Tuple[float, float, List[PIL.Image.Image]]], float]:
    if not video_path:
        raise gr.Error("Please upload a video before running inference.")
    if not _HAS_MOVIEPY:
        raise gr.Error("moviepy is required for video inputs. Install it via `pip install moviepy==1.0.3`.")
    _wait_for_upload_completion(video_path)

    result = _extract_segments_moviepy(
        video_path,
        n_frames=n_frames,
        segment_duration=segment_duration,
        max_segments=max_segments,
    )
    if result:
        return result
    return _extract_segments_imageio(
        video_path,
        n_frames=n_frames,
        segment_duration=segment_duration,
        max_segments=max_segments,
    )


def _extract_segments_moviepy(
    video_path: str,
    *,
    n_frames: int,
    segment_duration: int,
    max_segments: int,
) -> Optional[Tuple[List[Tuple[float, float, List[PIL.Image.Image]]], float]]:
    try:
        with VideoFileClip(video_path) as clip:
            if clip.duration is None or clip.duration <= 0:
                return None
            duration = float(clip.duration)
            fps = float(clip.fps or 0.0)
            segments: List[Tuple[float, float, List[PIL.Image.Image]]] = []
            start = 0.0
            while start < duration and len(segments) < max_segments:
                end = min(duration, start + segment_duration)
                frames = _sample_moviepy_frames(clip, start, end, n_frames, fps)
                if not frames:
                    return None
                segments.append((start, end, frames))
                start = end
            return segments, duration
    except OSError:
        return None


def _sample_moviepy_frames(
    clip: VideoFileClip,
    start: float,
    end: float,
    n_frames: int,
    fps: float,
) -> List[PIL.Image.Image]:
    window = max(end - start, 1e-3)
    available = max(1, int(math.ceil(window * (fps if fps > 0 else 25))))
    steps = max(1, min(n_frames, available))
    timestamps = torch.linspace(start, end, steps=steps).tolist()
    frames: List[PIL.Image.Image] = []
    for t in timestamps:
        try:
            frames.append(PIL.Image.fromarray(clip.get_frame(float(t))))
        except OSError:
            return []
    return frames


def _extract_segments_imageio(
    video_path: str,
    *,
    n_frames: int,
    segment_duration: int,
    max_segments: int,
) -> Tuple[List[Tuple[float, float, List[PIL.Image.Image]]], float]:
    try:
        import imageio
    except ImportError as exc:  # pragma: no cover
        raise gr.Error("imageio is required for fallback video decoding. Install it via `pip install imageio`.") from exc

    try:
        reader = imageio.get_reader(video_path, "ffmpeg")
    except Exception as exc:
        raise gr.Error("Failed to open video even after upload completed. Please re-encode as H.264 and retry.") from exc

    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or 0.0)
        duration = float(meta.get("duration") or 0.0)
        total_frames = int(meta.get("nframes") or 0)
        if duration <= 0 and fps > 0 and total_frames > 0:
            duration = total_frames / fps
        if (total_frames <= 0 or not math.isfinite(total_frames)) and fps > 0 and duration > 0:
            total_frames = int(duration * fps)
        if duration <= 0:
            raise gr.Error("Could not determine video duration. Please trim or re-encode the file and retry.")
        if fps <= 0:
            fps = 25.0
        segments: List[Tuple[float, float, List[PIL.Image.Image]]] = []
        start = 0.0
        while start < duration and len(segments) < max_segments:
            end = min(duration, start + segment_duration)
            start_idx = int(start * fps)
            end_idx = max(start_idx, int(end * fps))
            if total_frames > 0:
                end_idx = min(end_idx, total_frames - 1)
            span = max(1, end_idx - start_idx + 1)
            frame_count = max(1, min(n_frames, span))
            indices = np.linspace(start_idx, end_idx, num=frame_count, dtype=int)
            frames: List[PIL.Image.Image] = []
            for idx in indices:
                try:
                    frame = reader.get_data(max(0, int(idx)))
                except Exception as exc:
                    reader.close()
                    raise gr.Error(
                        "Failed to decode video frames via fallback reader. Please trim or re-encode the video."
                    ) from exc
                frames.append(PIL.Image.fromarray(frame))
            segments.append((start, end, frames))
            start = end
        reader.close()
    except Exception:
        reader.close()
        raise
    if not segments:
        raise gr.Error("Could not extract any frames from the uploaded video.")
    return segments, duration


def _wait_for_upload_completion(path: str, timeout: float = 15.0, poll_interval: float = 0.5) -> None:
    last_size = -1
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            current_size = os.path.getsize(path)
        except OSError:
            current_size = -1
        if current_size > 0 and current_size == last_size:
            return
        last_size = current_size
        time.sleep(poll_interval)
    raise gr.Error("Video upload is still in progress. Please wait a moment and retry.")


def _decorate_prompt(prompt: str, num_images: int, has_video: bool) -> str:
    placeholders = "".join(["<image>\n" for _ in range(num_images)])
    if has_video:
        placeholders += "<video>\n"
    return f"{placeholders}{prompt}" if prompt else placeholders.rstrip()


def _format_response(response: str, thinking: Optional[str]) -> str:
    if thinking:
        return f"**Thinking**\n```text\n{thinking}\n```\n\n**Response**\n{response}"
    return response


def _run_chat(
    prompt: str,
    *,
    images: Optional[List[PIL.Image.Image]] = None,
    videos: Optional[List[List[PIL.Image.Image]]] = None,
    do_sample: bool,
    max_new_tokens: int,
    enable_thinking: bool = False,
    thinking_budget: Optional[int] = None,
    min_pixels: int = 448 * 448,
    max_pixels: int = 1792 * 1792,
) -> str:
    model = _ensure_model(MODEL_PATH or "", DEVICE or "")
    visual_prompt = _decorate_prompt(prompt, len(images or []), videos is not None)
    response, thinking, _ = model.chat(
        prompt=visual_prompt,
        images=images,
        videos=videos,
        history=None,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return _format_response(response, thinking if enable_thinking else None)


def _require_prompt(prompt: str) -> None:
    if not prompt.strip():
        raise gr.Error("Prompt is required for this tab.")


def run_single_image(image: Optional[PIL.Image.Image], prompt: str, do_sample: bool, max_new_tokens: int) -> str:
    _require_prompt(prompt)
    if image is None:
        raise gr.Error("Please upload an image.")
    return _run_chat(
        prompt,
        images=[image],
        videos=None,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
    )


def run_multi_image(files: Optional[List[str]], prompt: str, do_sample: bool, max_new_tokens: int) -> str:
    _require_prompt(prompt)
    images = _files_to_images(files)
    if not images:
        raise gr.Error("Please upload at least one image.")
    return _run_chat(
        prompt,
        images=images,
        videos=None,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        max_pixels=896 * 896,
    )


def run_video(
    video_path: Optional[str],
    prompt: str,
    do_sample: bool,
    max_new_tokens: int,
    n_frames: int,
    segment_duration: int,
    max_segments: int,
) -> str:
    _require_prompt(prompt)
    segments, total_duration = _extract_video_segments(
        video_path,
        n_frames=n_frames,
        segment_duration=segment_duration,
        max_segments=max_segments,
    )
    if not segments:
        raise gr.Error("Unable to parse frames from the uploaded video.")

    outputs: List[str] = []
    for start, end, frames in segments:
        segment_prompt = f"{prompt}\n\n请基于视频时间段 {start:.1f}-{end:.1f} 秒进行回答。"
        response = _run_chat(
            segment_prompt,
            images=None,
            videos=[frames],
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            max_pixels=896 * 896,
        )
        outputs.append(f"**{start:.0f}-{end:.0f}s**\n{response}")

    if not outputs:
        raise gr.Error("Failed to generate any responses for the provided video.")

    processed_until = segments[-1][1]
    if processed_until + 1e-3 < total_duration:
        remaining = total_duration - processed_until
        outputs.append(
            f"_已分析前 {processed_until:.1f}s，剩余约 {remaining:.1f}s 可通过增加段数或调整段长继续处理。_"
        )
    return "\n\n".join(outputs)


def run_text_only(prompt: str, do_sample: bool, max_new_tokens: int) -> str:
    _require_prompt(prompt)
    return _run_chat(
        prompt,
        images=None,
        videos=None,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
    )


def run_thinking(
    image: Optional[PIL.Image.Image],
    prompt: str,
    do_sample: bool,
    max_new_tokens: int,
    enable_thinking: bool,
    thinking_budget: int,
) -> str:
    _require_prompt(prompt)
    if image is None:
        raise gr.Error("Please upload an image for reflective reasoning.")
    return _run_chat(
        prompt,
        images=[image],
        videos=None,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Ovis 2.5 Multi-Scenario Demo", theme=gr.themes.Default()) as demo:
        gr.Markdown("## Ovis 2.5 Demo")
        gr.Markdown(
            "Run single-image, multi-image, video, text-only, and reflective reasoning inference in one place."
        )

        with gr.Tab("Single Image"):
            gr.Markdown("Upload one image and provide a prompt.")
            image = gr.Image(type="pil", label="Image", height=300)
            prompt = gr.Textbox(label="Prompt", lines=3, value="请详细描述图片的内容")
            with gr.Accordion("Generation Options", open=False):
                do_sample = gr.Checkbox(label="Enable Sampling", value=False)
                max_tokens = gr.Slider(32, 2048, 1024, step=32, label="Max New Tokens")
            output = gr.Markdown()
            run_btn = gr.Button("Generate")
            run_btn.click(
                fn=run_single_image,
                inputs=[image, prompt, do_sample, max_tokens],
                outputs=output,
            )

        with gr.Tab("Multi Image"):
            gr.Markdown("Select multiple images to build richer prompts.")
            files = gr.Files(label="Images", file_types=["image"], file_count="multiple", type="filepath")
            prompt_m = gr.Textbox(label="Prompt", lines=3, value="这两张图是不是都在打架？")
            with gr.Accordion("Generation Options", open=False):
                do_sample_m = gr.Checkbox(label="Enable Sampling", value=False)
                max_tokens_m = gr.Slider(32, 2048, 1024, step=32, label="Max New Tokens")
            output_m = gr.Markdown()
            run_btn_m = gr.Button("Generate")
            run_btn_m.click(
                fn=run_multi_image,
                inputs=[files, prompt_m, do_sample_m, max_tokens_m],
                outputs=output_m,
            )

        with gr.Tab("Video"):
            gr.Markdown("Upload a short video (moviepy required).")
            video = gr.Video(label="Video", sources=["upload"])
            prompt_v = gr.Textbox(label="Prompt", lines=3, value="请描述视频的内容")
            with gr.Accordion("Generation Options", open=False):
                do_sample_v = gr.Checkbox(label="Enable Sampling", value=False)
                max_tokens_v = gr.Slider(32, 2048, 1024, step=32, label="Max New Tokens")
                frame_count = gr.Slider(4, 32, 8, step=1, label="Frames to Sample")
                segment_length = gr.Slider(10, 20, 10, step=1, label="Segment Length (s)")
                segment_limit = gr.Slider(1, 12, 6, step=1, label="Max Segments to Analyze")
            output_v = gr.Markdown()
            run_btn_v = gr.Button("Generate")
            run_btn_v.click(
                fn=run_video,
                inputs=[video, prompt_v, do_sample_v, max_tokens_v, frame_count, segment_length, segment_limit],
                outputs=output_v,
            )

        with gr.Tab("Text Only"):
            gr.Markdown("Pure text chat with the model.")
            prompt_t = gr.Textbox(label="Prompt", lines=4)
            with gr.Accordion("Generation Options", open=False):
                do_sample_t = gr.Checkbox(label="Enable Sampling", value=True)
                max_tokens_t = gr.Slider(32, 2048, 1024, step=32, label="Max New Tokens")
            output_t = gr.Markdown()
            run_btn_t = gr.Button("Generate")
            run_btn_t.click(
                fn=run_text_only,
                inputs=[prompt_t, do_sample_t, max_tokens_t],
                outputs=output_t,
            )

        with gr.Tab("Reflective Reasoning"):
            gr.Markdown("Enable thinking mode to inspect intermediate reasoning steps.")
            image_think = gr.Image(type="pil", label="Image", height=300)
            prompt_think = gr.Textbox(label="Prompt", lines=3, value="图中一个杯子有多高？")
            with gr.Accordion("Generation Options", open=True):
                do_sample_think = gr.Checkbox(label="Enable Sampling", value=True)
                max_tokens_think = gr.Slider(512, 16384, 4096, step=64, label="Max New Tokens")
                enable_think = gr.Checkbox(label="Enable Thinking Phase", value=True)
                thinking_budget = gr.Slider(256, 16384, 4096, step=64, label="Thinking Budget")
            output_think = gr.Markdown()
            run_btn_think = gr.Button("Generate")
            run_btn_think.click(
                fn=run_thinking,
                inputs=[
                    image_think,
                    prompt_think,
                    do_sample_think,
                    max_tokens_think,
                    enable_think,
                    thinking_budget,
                ],
                outputs=output_think,
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified demo for Ovis models.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "Ovis2.5-9B"),
        help="Path or hub name for the model.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to place the model on.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Host interface for Gradio.")
    return parser.parse_args()


def main() -> None:
    global MODEL_PATH, DEVICE
    args = parse_args()
    MODEL_PATH = args.model_path
    DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _ensure_model(MODEL_PATH, DEVICE)

    demo = build_ui()
    print(f"Launching app on http://{args.server_name}:{args.port}")
    demo.queue().launch(server_name=args.server_name, server_port=args.port, share=False, ssl_verify=False)


if __name__ == "__main__":
    main()
