#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio-powered FlashVSR v1.1 Web UI."""

import os
import sys
import tempfile
import threading
from types import ModuleType
from typing import Callable, Dict, List, Tuple

import gradio as gr
import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WANVSR_DIR = os.path.join(ROOT_DIR, "examples", "WanVSR")
if WANVSR_DIR not in sys.path:
    sys.path.append(WANVSR_DIR)

DEMO_INPUT_DIR = os.path.join(WANVSR_DIR, "inputs")
PROMPT_TENSOR_PATH = os.path.join(WANVSR_DIR, "prompt_tensor", "posi_prompt.pth")


def _discover_demo_examples() -> List[str]:
    if not os.path.isdir(DEMO_INPUT_DIR):
        return []
    exts = (".mp4", ".mov", ".avi", ".mkv")
    files = [
        os.path.join(DEMO_INPUT_DIR, f)
        for f in os.listdir(DEMO_INPUT_DIR)
        if f.lower().endswith(exts)
    ]
    files.sort()
    return files


DEMO_EXAMPLE_PATHS = _discover_demo_examples()

def _ensure_modelscope_stub():
    """Provide a lightweight stub so importing diffsynth doesn't require modelscope."""

    try:
        __import__("modelscope")
        return
    except ModuleNotFoundError:
        pass

    stub = ModuleType("modelscope")

    def _missing_snapshot_download(*_, **__):
        raise ImportError(
            "modelscope is not installed. Install it via `pip install modelscope` if you "
            "plan to download weights automatically. Existing local checkpoints work without it."
        )

    stub.snapshot_download = _missing_snapshot_download
    sys.modules["modelscope"] = stub


_ensure_modelscope_stub()


def _resolve_checkpoint_dir() -> str:
    """Return the FlashVSR-v1.1 checkpoint folder, supporting both legacy and new locations."""

    candidates = [
        os.path.join(ROOT_DIR, "checkpoints", "FlashVSR-v1.1"),
        os.path.join(ROOT_DIR, "Block-Sparse-Attention", "checkpoints", "FlashVSR-v1.1"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]  # default to root-level path for error reporting


from diffsynth import (  # noqa: E402
    FlashVSRFullPipeline,
    FlashVSRTinyLongPipeline,
    FlashVSRTinyPipeline,
    ModelManager,
)
from utils.TCDecoder import build_tcdecoder  # noqa: E402
from utils.utils import Causal_LQ4x_Proj  # noqa: E402


CHECKPOINT_DIR = _resolve_checkpoint_dir()
DEMO_INPUT_DIR = os.path.join(WANVSR_DIR, "inputs")
LQ_PROJ_PATH = os.path.join(CHECKPOINT_DIR, "LQ_proj_in.ckpt")
TC_DECODER_PATH = os.path.join(CHECKPOINT_DIR, "TCDecoder.ckpt")

VARIANT_CONFIG: Dict[str, Dict] = {
    "full": {
        "label": "FlashVSR v1.1 Full",
        "pipeline_cls": FlashVSRFullPipeline,
        "model_paths": [
            os.path.join(CHECKPOINT_DIR, "diffusion_pytorch_model_streaming_dmd.safetensors"),
            os.path.join(CHECKPOINT_DIR, "Wan2.1_VAE.pth"),
        ],
        "preprocess_device": "cuda",
        "supports_tiling": True,
        "needs_tcdecoder": False,
    }
}

DEFAULT_VARIANT_KEY = "full"

PIPELINES: Dict[str, object] = {}
PIPELINE_LOCKS: Dict[str, threading.Lock] = {k: threading.Lock() for k in VARIANT_CONFIG}


# ----------------------------- Utility helpers ----------------------------- #


def _extract_input_path(file_input) -> str:
    """Normalize a Gradio File/Video input to a local path."""

    if not file_input:
        return ""
    if isinstance(file_input, str):
        return file_input
    if isinstance(file_input, dict):
        for key in ("path", "name"):
            path = file_input.get(key)
            if isinstance(path, str) and os.path.exists(path):
                return path
        data = file_input.get("data")
        if isinstance(data, str) and os.path.exists(data):
            return data
    if hasattr(file_input, "name"):
        return getattr(file_input, "name")
    return ""

def _log(msg: str, logger: List[str]):
    logger.append(msg)
    print(msg)


def tensor2video(frames: torch.Tensor) -> List[Image.Image]:
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames]


def natural_key(name: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def list_images_natural(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def discover_demo_inputs():
    if not os.path.isdir(DEMO_INPUT_DIR):
        return []
    exts = (".mp4", ".mov", ".avi", ".mkv")
    files = [f for f in os.listdir(DEMO_INPUT_DIR) if f.lower().endswith(exts)]
    files.sort(key=natural_key)
    return [(f, os.path.join(DEMO_INPUT_DIR, f)) for f in files]


DEMO_INPUTS = discover_demo_inputs()
DEMO_EXAMPLES = [[path] for _, path in DEMO_INPUTS]


def largest_8n1_leq(n: int) -> int:
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def is_video(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device: str = "cuda"):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    up = img.resize((sW, sH), Image.BICUBIC)
    lft = max(0, (sW - tW) // 2)
    top = max(0, (sH - tH) // 2)
    return up.crop((lft, top, lft + tW, top + tH))


def prepare_input_tensor(
    path: str,
    scale: float = 4.0,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    logger: Callable[[str], None] | None = None,
):
    log = logger or (lambda msg: None)

    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        log(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Frames: {N0}")
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale)
        log(f"[{os.path.basename(path)}] Scaled x{scale:.2f}: {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}")
        paths = paths[:F]
        log(f"[{os.path.basename(path)}] Target Frames (8n-3): {F - 4}")

        frames = []
        for p in paths:
            with Image.open(p).convert("RGB") as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        fps = 30
        return vid, tH, tW, F, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert("RGB")
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(reader):
            try:
                nf = meta.get("nframes", None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return reader.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        reader.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        basename = os.path.basename(path)
        log(f"[{basename}] Original Resolution: {w0}x{h0} | Frames: {total} | FPS: {fps}")
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale)
        log(f"[{basename}] Scaled x{scale:.2f}: {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}")
        idx = idx[:F]
        log(f"[{basename}] Target Frames (8n-3): {F - 4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert("RGB")
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        return vid, tH, tW, F, fps

    raise ValueError(f"Unsupported input: {path}")


def save_video(frames: List[Image.Image], save_path: str, fps: int = 30, quality: int = 6):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()


# --------------------------- Pipeline construction ------------------------- #

def _ensure_files_exist(paths: List[str]):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required weights: " + ", ".join(os.path.relpath(p, ROOT_DIR) for p in missing)
        )


def load_pipeline(variant_key: str, logger: List[str]):
    if variant_key in PIPELINES:
        return PIPELINES[variant_key]

    lock = PIPELINE_LOCKS[variant_key]
    with lock:
        if variant_key in PIPELINES:
            return PIPELINES[variant_key]

        cfg = VARIANT_CONFIG[variant_key]
        _log(f"Loading {cfg['label']} pipeline...", logger)
        _ensure_files_exist(cfg["model_paths"] + [LQ_PROJ_PATH])
        if cfg["needs_tcdecoder"]:
            _ensure_files_exist([TC_DECODER_PATH])

        mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        mm.load_models(cfg["model_paths"])
        pipe = cfg["pipeline_cls"].from_model_manager(mm, device="cuda")

        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
            "cuda", dtype=torch.bfloat16
        )
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_PROJ_PATH, map_location="cpu"), strict=True)
        pipe.denoising_model().LQ_proj_in.to("cuda")

        if cfg["needs_tcdecoder"]:
            pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
            pipe.TCDecoder.load_state_dict(torch.load(TC_DECODER_PATH, map_location="cpu"), strict=False)

        if isinstance(pipe, FlashVSRFullPipeline):
            # Match inference script optimizations
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None

        pipe.to("cuda")
        pipe.enable_vram_management(num_persistent_param_in_dit=None)

        if not os.path.exists(PROMPT_TENSOR_PATH):
            raise FileNotFoundError(
                "Missing prompt tensor at examples/WanVSR/prompt_tensor/posi_prompt.pth."
                " Please pull the assets from the official repo/HF release."
            )
        prompt_tensor = torch.load(PROMPT_TENSOR_PATH, map_location="cpu")

        pipe.init_cross_kv(context_tensor=prompt_tensor)
        pipe.load_models_to_device(["dit", "vae"])

        PIPELINES[variant_key] = pipe
        _log(f"{cfg['label']} is ready.", logger)
        return pipe


def preload_pipeline():
    logs: List[str] = []
    try:
        _log("正在预加载 FlashVSR v1.1 Full 模型...", logs)
        load_pipeline(DEFAULT_VARIANT_KEY, logs)
        _log("模型预加载完成，可直接推理。", logs)
    except Exception as exc:
        raise RuntimeError(
            "初始化 FlashVSR 模型失败，请检查权重是否完整且路径正确。详情: " + str(exc)
        ) from exc


# ------------------------------ Inference API ------------------------------ #

def run_inference(
    input_file,
    scale: float,
    sparse_ratio: float,
    local_range: int,
    seed: int,
    tiled: bool,
    color_fix: bool,
):
    input_video = _extract_input_path(input_file)
    if not input_video:
        raise gr.Error("Please upload a video file to process.")
    if not torch.cuda.is_available():
        raise gr.Error("FlashVSR requires a CUDA-enabled GPU.")

    variant_key = DEFAULT_VARIANT_KEY
    cfg = VARIANT_CONFIG[variant_key]
    logs: List[str] = []

    _log("Clearing CUDA cache...", logs)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    preprocess_device = cfg["preprocess_device"] if torch.cuda.is_available() else "cpu"
    LQ, th, tw, F, fps = prepare_input_tensor(
        input_video,
        scale=scale,
        dtype=torch.bfloat16,
        device=preprocess_device,
        logger=lambda msg: _log(msg, logs),
    )

    if preprocess_device != "cuda":
        LQ = LQ.to("cuda")

    pipe = load_pipeline(variant_key, logs)

    topk_ratio = float(sparse_ratio) * 768 * 1280 / (th * tw)
    call_kwargs = dict(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=int(seed),
        LQ_video=LQ,
        num_frames=F,
        height=th,
        width=tw,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=topk_ratio,
        kv_ratio=3.0,
        local_range=int(local_range),
        color_fix=bool(color_fix),
    )
    if cfg["supports_tiling"]:
        call_kwargs["tiled"] = bool(tiled)

    _log("Running FlashVSR...", logs)
    video = pipe(**call_kwargs)
    frames = tensor2video(video)

    tmp_dir = tempfile.mkdtemp(prefix="flashvsr_")
    output_path = os.path.join(tmp_dir, f"{variant_key}_upscaled.mp4")
    save_video(frames, output_path, fps=fps, quality=6)
    _log(f"Saved output to {output_path}", logs)

    return output_path, "\n".join(logs)


# --------------------------------- Gradio UI -------------------------------- #

def build_demo():
    with gr.Blocks(title="FlashVSR v1.1 WebUI", css="footer {visibility: hidden}") as demo:
        gr.Markdown(
            """
            ## FlashVSR v1.1 WebUI
            上传低清视频，一键调用官方 **FlashVSR v1.1 Full** 单步推理管线，获得最佳画质的 4× 超分结果。
            请确认 v1.1 权重位于仓库根目录下的 `checkpoints/FlashVSR-v1.1/`。
            """
        )
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    label="低清视频输入",
                    height=360,
                    autoplay=False,
                )
                if DEMO_EXAMPLE_PATHS:
                    gr.Examples(
                        label="内置示例视频 (examples/WanVSR/inputs)",
                        examples=[[p] for p in DEMO_EXAMPLE_PATHS],
                        inputs=[video_input],
                        cache_examples=False,
                        examples_per_page=4,
                    )
                gr.Markdown(
                    "**当前模型：** FlashVSR v1.1 Full（最高画质，包含 Wan2.1 VAE 与 diffusion 主干）"
                )
                scale_slider = gr.Slider(
                    label="预上采样倍率",
                    minimum=3.5,
                    maximum=4.5,
                    step=0.25,
                    value=4.0,
                )
                sparse_slider = gr.Slider(
                    label="稀疏比 (top-k)",
                    minimum=1.0,
                    maximum=3.0,
                    step=0.1,
                    value=2.0,
                    info="1.5 更快，2.0 更稳定",
                )
                local_slider = gr.Slider(
                    label="局部窗口大小",
                    minimum=7,
                    maximum=13,
                    step=2,
                    value=11,
                )
                seed_number = gr.Number(label="随机种子", value=0, precision=0)
                tiled_checkbox = gr.Checkbox(
                    label="启用分块推理（仅 Full 有效）",
                    value=False,
                )
                color_fix_checkbox = gr.Checkbox(label="开启色彩修复", value=True)
                submit_btn = gr.Button("开始推理", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="超分输出", height=360)
                log_output = gr.Textbox(label="日志输出", lines=14)

        submit_btn.click(
            run_inference,
            inputs=[
                video_input,
                scale_slider,
                sparse_slider,
                local_slider,
                seed_number,
                tiled_checkbox,
                color_fix_checkbox,
            ],
            outputs=[output_video, log_output],
        )
    return demo


def main():
    preload_pipeline()
    demo = build_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
