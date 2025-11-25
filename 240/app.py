#!/usr/bin/env python3
"""Standalone Gradio UI that embeds the Vivid-VR inference pipeline."""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXVividVRControlNetModel,
    CogVideoXVividVRControlNetPipeline,
    CogVideoXVividVRTransformer3DModel,
)
from transformers import T5EncoderModel

from VRDiT.cogvlm2 import CogVLM2_Captioner
from VRDiT.colorfix import adaptive_instance_normalization
from VRDiT.inference import infer_split_clips, infer_whole_video
from VRDiT.utils import export_to_video, free_memory, load_video

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
COGVIDEOX_DIR = CKPT_DIR / "CogVideoX1.5-5B"
COGVLM2_DIR = CKPT_DIR / "cogvlm2-llama3-caption"
VIVID_VR_ROOT_DIR = CKPT_DIR / "Vivid-VR"
EASYOCR_DIR = CKPT_DIR / "easyocr"
REALESRGAN_CKPT = CKPT_DIR / "RealESRGAN" / "RealESRGAN_x2plus.pth"
APP_OUTPUT_DIR = ROOT / "app_outputs"
APP_OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_TILE_SIZE = 128

_MODEL_BUNDLES: dict[str, dict] = {}
_MODEL_LOCK = threading.Lock()
_TEXT_FIXER = None
_TEXT_FIXER_LOCK = threading.Lock()
VIVID_VR_WEIGHTS_DIR: Path | None = None


def _format_missing(paths: list[Path]) -> str:
    return "\n".join(f"- {p}" for p in paths)


def _resolve_vivid_vr_weights() -> Path:
    global VIVID_VR_WEIGHTS_DIR
    if VIVID_VR_WEIGHTS_DIR is not None and VIVID_VR_WEIGHTS_DIR.exists():
        return VIVID_VR_WEIGHTS_DIR

    candidates: list[Path] = []
    env_path = os.environ.get("VIVID_VR_WEIGHTS")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())
    candidates.extend([
        VIVID_VR_ROOT_DIR,
        VIVID_VR_ROOT_DIR / "ckpts" / "Vivid-VR",
        CKPT_DIR / "ckpts" / "Vivid-VR",
    ])

    required_files = [
        "connectors.pt",
        "control_feat_proj.pt",
        "control_patch_embed.pt",
        "controlnet",
    ]

    for candidate in candidates:
        candidate = candidate.resolve()
        missing = [candidate / file for file in required_files if not (candidate / file).exists()]
        if not missing:
            VIVID_VR_WEIGHTS_DIR = candidate
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "未找到 Vivid-VR 权重文件，请确认下列目录之一包含 connectors.pt 等文件:\n" + searched
    )


def _validate_paths(textfix: bool) -> None:
    missing = [p for p in (CKPT_DIR, COGVIDEOX_DIR, COGVLM2_DIR) if not p.exists()]
    if missing:
        raise gr.Error("缺少以下必要权重，请放到 checkpoints/ 后重试:\n" + _format_missing(missing))

    try:
        _resolve_vivid_vr_weights()
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc

    if textfix:
        extras = [p for p in (EASYOCR_DIR, REALESRGAN_CKPT) if not p.exists()]
        if extras:
            raise gr.Error("TextFix 需要以下文件：\n" + _format_missing(extras))


def _ensure_cuda(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise gr.Error("未检测到可用的 CUDA GPU，无法在 cuda 上运行。")


def _collect_video_info(video_path: Path) -> dict:
    if not video_path.exists():
        raise gr.Error(f"无法找到输入视频：{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error(f"无法读取视频：{video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    cap.release()

    fps = max(fps, 1.0)
    return {"path": str(video_path), "width": width, "height": height, "fps": fps}


def _build_args(
    guidance_scale: float,
    restoration_guidance_scale: float,
    num_inference_steps: int,
    num_temporal_process_frames: int,
    seed: int,
    device: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        guidance_scale=guidance_scale,
        restoration_guidance_scale=restoration_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_temporal_process_frames=num_temporal_process_frames,
        tile_size=DEFAULT_TILE_SIZE,
        use_dynamic_cfg=False,
        device=device,
        seed=seed,
    )


def _load_model_bundle(device: str) -> dict:
    print("[Vivid-VR] Loading models… this may take a while.")

    vivid_vr_weights = _resolve_vivid_vr_weights()

    captioner_model = CogVLM2_Captioner(model_path=str(COGVLM2_DIR))

    text_encoder = T5EncoderModel.from_pretrained(str(COGVIDEOX_DIR), subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(dtype=torch.bfloat16)

    transformer = CogVideoXVividVRTransformer3DModel.from_pretrained(
        str(COGVIDEOX_DIR), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    transformer.requires_grad_(False)
    transformer.to(dtype=torch.bfloat16)
    transformer.patch_embed.use_positional_embeddings = False
    transformer.patch_embed.use_learned_positional_embeddings = False
    transformer.config.use_learned_positional_embeddings = False
    transformer.config.use_rotary_positional_embeddings = True

    vae = AutoencoderKLCogVideoX.from_pretrained(str(COGVIDEOX_DIR), subfolder="vae")
    vae.requires_grad_(False)
    vae.to(dtype=torch.bfloat16)
    vae.enable_slicing()
    vae.enable_tiling()

    controlnet = CogVideoXVividVRControlNetModel.from_transformer(transformer=transformer, num_layers=6)
    controlnet.requires_grad_(False)
    controlnet.to(dtype=torch.bfloat16)

    scheduler = CogVideoXDPMScheduler.from_pretrained(str(COGVIDEOX_DIR), subfolder="scheduler")

    transformer.connectors.load_state_dict(torch.load(vivid_vr_weights / "connectors.pt", map_location="cpu"))
    transformer.control_feat_proj.load_state_dict(torch.load(vivid_vr_weights / "control_feat_proj.pt", map_location="cpu"))
    transformer.control_patch_embed.load_state_dict(torch.load(vivid_vr_weights / "control_patch_embed.pt", map_location="cpu"))
    load_model = CogVideoXVividVRControlNetModel.from_pretrained(str(vivid_vr_weights), subfolder="controlnet")
    controlnet.register_to_config(**load_model.config)
    controlnet.load_state_dict(load_model.state_dict())
    del load_model

    pipe = CogVideoXVividVRControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=str(COGVIDEOX_DIR),
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        controlnet=controlnet,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)

    bundle = {
        "captioner_model": captioner_model,
        "pipe": pipe,
        "vae_scale_factor_spatial": 2 ** (len(vae.config.block_out_channels) - 1),
        "vae_scale_factor_temporal": vae.config.temporal_compression_ratio,
    }
    print("[Vivid-VR] Models loaded.")
    return bundle


def _get_model_bundle(device: str) -> dict:
    with _MODEL_LOCK:
        bundle = _MODEL_BUNDLES.get(device)
        if bundle is not None:
            return bundle

    bundle = _load_model_bundle(device)
    with _MODEL_LOCK:
        _MODEL_BUNDLES[device] = bundle
    return bundle


def _get_text_fixer():
    from VRDiT.textfix import TextFixer  # lazy import to avoid loading easyocr when unused

    global _TEXT_FIXER
    with _TEXT_FIXER_LOCK:
        if _TEXT_FIXER is None:
            _TEXT_FIXER = TextFixer(easyocr_model_path=str(EASYOCR_DIR), enhancer_model_path=str(REALESRGAN_CKPT))
    return _TEXT_FIXER


def run_vivid_vr(
    input_video: str | None,
    num_temporal_process_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    restoration_guidance_scale: float,
    upscale: float,
    seed: int,
    device: str,
    textfix: bool,
    save_images: bool,
) -> tuple[str, str]:
    if input_video is None:
        raise gr.Error("请先上传一个输入视频。")

    num_temporal_process_frames = int(num_temporal_process_frames)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    restoration_guidance_scale = float(restoration_guidance_scale)
    upscale = float(upscale)
    seed = int(seed)

    _validate_paths(textfix)
    _ensure_cuda(device)

    bundle = _get_model_bundle(device)
    captioner_model = bundle["captioner_model"]
    pipe = bundle["pipe"]
    vae_scale_factor_spatial = bundle["vae_scale_factor_spatial"]
    vae_scale_factor_temporal = bundle["vae_scale_factor_temporal"]

    input_path = Path(input_video)
    info = _collect_video_info(input_path)
    control_video = load_video(str(input_path))

    if upscale == 0.0:
        scale_factor = 1024.0 / min(control_video.size(2), control_video.size(3))
        control_video = F.interpolate(control_video, scale_factor=scale_factor, mode='bicubic').clip(0, 1)
        info['height'], info['width'] = control_video.size(2), control_video.size(3)
    elif upscale != 1.0:
        control_video = F.interpolate(control_video, scale_factor=upscale, mode='bicubic').clip(0, 1)
        info['height'], info['width'] = control_video.size(2), control_video.size(3)

    gen_height = 8 * math.ceil(info['height'] / 8) if info['height'] < DEFAULT_TILE_SIZE * vae_scale_factor_spatial else info['height']
    gen_width = 8 * math.ceil(info['width'] / 8) if info['width'] < DEFAULT_TILE_SIZE * vae_scale_factor_spatial else info['width']

    args = _build_args(
        guidance_scale=guidance_scale,
        restoration_guidance_scale=restoration_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_temporal_process_frames=num_temporal_process_frames,
        seed=seed,
        device=device,
    )

    with torch.inference_mode():
        if control_video.size(0) > num_temporal_process_frames:
            video = infer_split_clips(
                args=args,
                captioner_model=captioner_model,
                pipe=pipe,
                info=info,
                control_video=control_video,
                gen_height=gen_height,
                gen_width=gen_width,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                vae_scale_factor_temporal=vae_scale_factor_temporal,
            )
        else:
            video = infer_whole_video(
                args=args,
                captioner_model=captioner_model,
                pipe=pipe,
                info=info,
                control_video=control_video,
                gen_height=gen_height,
                gen_width=gen_width,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
            )

        reference_video = control_video.to(device)
        samples = adaptive_instance_normalization(
            torch.from_numpy(video).permute(0, 3, 1, 2).to(device),
            reference_video,
        )

        if textfix:
            text_fixer = _get_text_fixer()
            samples = text_fixer(video=samples, ref_video=reference_video, device=device)

        samples = samples.cpu().clip(0, 1).permute(0, 2, 3, 1).float().numpy()
        del reference_video

    timestamp = int(time.time())
    output_path = APP_OUTPUT_DIR / f"vividvr_{timestamp}_{input_path.stem}.mp4"
    export_to_video(samples, str(output_path), fps=info['fps'])

    if save_images:
        image_dir = APP_OUTPUT_DIR / "images" / f"{timestamp}_{input_path.stem}"
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(samples):
            Image.fromarray((frame * 255).astype(np.uint8)).save(image_dir / f"{idx:06d}.png")

    free_memory()

    log_lines = [
        f"Input: {input_path}",
        f"Frames: {control_video.size(0)}, Resolution after preprocess: {info['height']}x{info['width']}",
        f"Guidance scale: {guidance_scale}",
        f"Temporal window: {num_temporal_process_frames}",
        f"Inference steps: {num_inference_steps}",
        f"Restoration guidance scale: {restoration_guidance_scale}",
    ]
    if textfix:
        log_lines.append("TextFix enabled (EasyOCR + Real-ESRGAN)")
    if save_images:
        log_lines.append(f"Saved frames to: {image_dir}")

    return str(output_path), "\n".join(log_lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Vivid-VR Gradio UI") as demo:
        gr.Markdown(
            """
            # Vivid-VR 视频增强

            该界面直接调用官方 `VRDiT/inference.py` 中的推理逻辑（已嵌入当前脚本，无需额外命令行）。
            请确认所有模型权重位于 `checkpoints/` 目录下。
            """
        )

        with gr.Row():
            input_video = gr.Video(label="输入视频", sources=["upload"], height=360)
            output_video = gr.Video(label="输出视频", height=360)

        with gr.Accordion("高级参数", open=False):
            num_temporal_process_frames = gr.Slider(
                value=121,
                minimum=9,
                maximum=257,
                step=8,
                label="num_temporal_process_frames (需满足 8k+1)",
            )
            num_inference_steps = gr.Slider(value=50, minimum=10, maximum=75, step=1, label="num_inference_steps")
            guidance_scale = gr.Slider(value=6.0, minimum=1.0, maximum=12.0, step=0.5, label="guidance_scale")
            restoration_guidance_scale = gr.Slider(
                value=-1.0,
                minimum=-1.0,
                maximum=5.0,
                step=0.1,
                label="restoration_guidance_scale",
            )
            upscale = gr.Radio(
                choices=[0.0, 1.0, 1.5, 2.0],
                value=0.0,
                label="upscale (0 表示短边拉至 1024)",
            )
            seed = gr.Number(value=42, label="seed", precision=0)
            device = gr.Radio(choices=["cuda"], value="cuda", label="device")
            textfix = gr.Checkbox(label="textfix", value=False)
            save_images = gr.Checkbox(label="save_images", value=False)

        logs = gr.Textbox(label="推理日志", lines=12, interactive=False)
        run_button = gr.Button("开始推理", variant="primary")

        run_button.click(
            fn=run_vivid_vr,
            inputs=[
                input_video,
                num_temporal_process_frames,
                num_inference_steps,
                guidance_scale,
                restoration_guidance_scale,
                upscale,
                seed,
                device,
                textfix,
                save_images,
            ],
            outputs=[output_video, logs],
        )

        demo.queue(max_size=1)

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
