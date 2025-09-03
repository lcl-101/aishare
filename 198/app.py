"""Gradio WebUI for Stand-In with 4 tabs:
1. Base IP2V generation (infer.py)
2. IP2V + LoRA stylization (infer_with_lora.py)
3. Face Swap (infer_face_swap.py)
4. VACE Motion / Reference (infer_with_vace.py)

Models are lazily loaded & cached per mode. LoRA pipelines are cached by (path, scale).
"""

import os
import time
import tempfile
import copy
from typing import Tuple, Dict
import torch
import gradio as gr
from PIL import Image

from data.video import save_video
from wan_loader import load_wan_pipe
from models.set_condition_branch import set_stand_in
from preprocessor import FaceProcessor, VideoMaskGenerator

DEVICE = os.environ.get("STANDIN_DEVICE", "cuda")
BASE_MODEL_PATH = os.environ.get("STANDIN_BASE", "checkpoints/base_model/")
STAND_IN_PATH = os.environ.get(
    "STANDIN_STANDIN", "checkpoints/Stand-In/Stand-In_wan2.1_T2V_14B_ver1.0.ckpt"
)
ANTELOPEV2_PATH = os.environ.get("STANDIN_ANTELOPE", "checkpoints/antelopev2")
VACE_PATH = os.environ.get("STANDIN_VACE", "checkpoints/VACE/")

assert os.path.isdir(BASE_MODEL_PATH), f"Missing base model at {BASE_MODEL_PATH}"
assert os.path.isdir(ANTELOPEV2_PATH), f"Missing antelopev2 at {ANTELOPEV2_PATH}"
assert os.path.isfile(STAND_IN_PATH), f"Missing Stand-In checkpoint at {STAND_IN_PATH}"

face_processor = FaceProcessor(antelopv2_path=ANTELOPEV2_PATH)
videomask_generator = None  # lazy

_cache: Dict[str, any] = {}
_lora_cache: Dict[Tuple[str, float], any] = {}

SINGLE_ACTIVE = os.environ.get("STANDIN_SINGLE_ACTIVE", "1") == "1"


def _prune_pipelines(retain: set[str] | None = None, retain_lora: set[Tuple[str, float]] | None = None):
    """Free pipelines not in retain sets to reduce RAM/VRAM pressure.

    Called automatically when STANDIN_SINGLE_ACTIVE=1 before instantiating a new heavy pipeline.
    """
    if not SINGLE_ACTIVE:
        return
    retain = retain or set()
    retain_lora = retain_lora or set()
    removed = []
    for k in list(_cache.keys()):
        if k not in retain:
            try:
                del _cache[k]
                removed.append(k)
            except Exception:
                pass
    for k in list(_lora_cache.keys()):
        if k not in retain_lora:
            try:
                del _lora_cache[k]
            except Exception:
                pass
    if removed:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[Stand-In] Released pipelines: {removed}")


def _resource_snapshot(label: str):
    if os.environ.get("STANDIN_PROFILE", "0") != "1":
        return
    rss_kb = None
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_kb = int(parts[1])
                    break
    except Exception:
        pass
    gpu_alloc = gpu_reserved = None
    if torch.cuda.is_available():
        try:
            gpu_alloc = torch.cuda.memory_allocated() / (1024**2)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
        except Exception:
            pass
    print(f"[PROFILE] {label} | RSS={rss_kb}KB | GPU alloc={gpu_alloc}MB | GPU reserved={gpu_reserved}MB")


def _base_pipe():
    if "base" not in _cache:
        _resource_snapshot("before_load_base")
        if SINGLE_ACTIVE:
            _prune_pipelines(retain={"base"})
        pipe = load_wan_pipe(base_path=BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
        set_stand_in(pipe, model_path=STAND_IN_PATH)
        _cache["base"] = pipe
        _resource_snapshot("after_load_base")
    return _cache["base"]


def _lora_pipe(lora_path: str, lora_scale: float):
    """Return a LoRA-applied pipeline without reloading base weights from disk each time.

    Strategy:
    - Reuse already loaded base pipeline (loads from disk only once for base).
    - Deep copy the base pipeline the first time we see a (lora_path, scale) pair, then apply LoRA.
      (Deep copy avoids permanently mutating the shared base pipeline and prevents LoRA stacking.)
    - Cache by (path, scale) so subsequent inferences skip both disk IO and LoRA re-application.
    """
    key = (lora_path, float(lora_scale))
    if key in _lora_cache:
        if SINGLE_ACTIVE:
            _prune_pipelines(retain_lora={key})
        return _lora_cache[key]
    if not os.path.isfile(lora_path):
        raise gr.Error(f"LoRA file not found: {lora_path}")
    base = _base_pipe()
    if SINGLE_ACTIVE:
        # In single-active mode, prefer re-loading a fresh base to avoid keeping both copies.
        # We load a new base pipeline (not deep copy) to save peak RAM (at cost of disk IO once).
        _resource_snapshot("before_load_lora_fresh_base")
        _prune_pipelines()  # remove existing pipelines including base
        base = load_wan_pipe(base_path=BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
        set_stand_in(base, model_path=STAND_IN_PATH)
        pipe = base
        _resource_snapshot("after_load_lora_fresh_base")
    else:
        # Deep copy to isolate LoRA modifications when multi-pipeline mode.
        _resource_snapshot("before_deepcopy_base_for_lora")
        pipe = copy.deepcopy(base)
        _resource_snapshot("after_deepcopy_base_for_lora")
    updated = pipe.load_lora(pipe.dit, lora_path, alpha=lora_scale)
    if updated == 0:
        raise gr.Error("LoRA matched 0 layers. Possibly incompatible.")
    _lora_cache[key] = pipe
    if SINGLE_ACTIVE:
        _prune_pipelines(retain_lora={key})
    return pipe


def _face_swap_pipe():
    if "face_swap" not in _cache:
        _resource_snapshot("before_load_face_swap")
        if SINGLE_ACTIVE:
            _prune_pipelines(retain={"face_swap"})
        pipe = load_wan_pipe(
            base_path=BASE_MODEL_PATH, face_swap=True, torch_dtype=torch.bfloat16
        )
        set_stand_in(pipe, model_path=STAND_IN_PATH)
        _cache["face_swap"] = pipe
        _resource_snapshot("after_load_face_swap")
    return _cache["face_swap"]


def _vace_pipe():
    if "vace" not in _cache:
        _resource_snapshot("before_load_vace")
        if SINGLE_ACTIVE:
            _prune_pipelines(retain={"vace"})
        pipe = load_wan_pipe(
            base_path=VACE_PATH, use_vace=True, torch_dtype=torch.bfloat16
        )
        set_stand_in(pipe, model_path=STAND_IN_PATH)
        _cache["vace"] = pipe
        _resource_snapshot("after_load_vace")
    return _cache["vace"]


def ui_base(ip_image: Image.Image, prompt: str, negative_prompt: str, seed: int, steps: int, fps: int, quality: int):
    if ip_image is None:
        raise gr.Error("Please upload a face image.")
    pipe = _base_pipe()
    ip = face_processor.process(ip_image)
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        ip_image=ip,
        num_inference_steps=int(steps),
        tiled=False,
    )
    return _save_temp(video, fps, quality)


def ui_lora(ip_image: Image.Image, prompt: str, negative_prompt: str, seed: int, steps: int, lora_path: str, lora_scale: float, fps: int, quality: int):
    if ip_image is None:
        raise gr.Error("Please upload a face image.")
    pipe = _lora_pipe(lora_path, lora_scale)
    ip = face_processor.process(ip_image)
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        ip_image=ip,
        num_inference_steps=int(steps),
        tiled=False,
    )
    return _save_temp(video, fps, quality)


def ui_face_swap(ip_image: Image.Image, input_video: str, prompt: str, negative_prompt: str, seed: int, steps: int, denoise: float, force_bg: bool, fps: int, quality: int):
    if ip_image is None or not input_video:
        raise gr.Error("Need face image and input video.")
    global videomask_generator
    if videomask_generator is None:
        videomask_generator = VideoMaskGenerator(antelopv2_path=ANTELOPEV2_PATH)
    pipe = _face_swap_pipe()
    ip_tensor, ip_rgba = face_processor.process(ip_image, extra_input=True)
    vid_tensor, face_mask, width, height, num_frames = videomask_generator.process(
        input_video, ip_rgba, random_horizontal_flip_chance=0.05, dilation_kernel_size=10
    )
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        width=width,
        height=height,
        num_frames=num_frames,
        denoising_strength=float(denoise),
        ip_image=ip_tensor,
        face_mask=face_mask,
        input_video=vid_tensor,
        num_inference_steps=int(steps),
        tiled=False,
        force_background_consistency=bool(force_bg),
    )
    return _save_temp(video, fps, quality)


def ui_vace(ip_image: Image.Image, reference_video: str, reference_image: Image.Image, prompt: str, negative_prompt: str, seed: int, steps: int, vace_scale: float, fps: int, quality: int):
    if ip_image is None or not reference_video:
        raise gr.Error("Need ip_image and reference_video.")
    _resource_snapshot("before_vace_infer")
    pipe = _vace_pipe()
    ip = face_processor.process(ip_image)
    # reference_image path may be required; if user provided an image file, save temp
    ref_img_path = None
    if isinstance(reference_image, Image.Image):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        reference_image.save(tmp.name)
        ref_img_path = tmp.name
    else:
        ref_img_path = reference_image
    video = pipe(
        prompt=prompt,
        vace_video=reference_video,
        vace_reference_image=ref_img_path,
        negative_prompt=negative_prompt,
        vace_scale=float(vace_scale),
        seed=int(seed),
        ip_image=ip,
        num_inference_steps=int(steps),
        tiled=False,
    )
    out = _save_temp(video, fps, quality)
    _resource_snapshot("after_vace_infer")
    return out


def _save_temp(video, fps: int, quality: int):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    save_video(video, path, fps=int(fps), quality=int(quality))
    return path


def _preview_video(path: str):
    if not path:
        raise gr.Error("Please provide a video path.")
    if not os.path.isfile(path):
        raise gr.Error(f"Video not found: {path}")
    return path


NEGATIVE_DEFAULT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

def _maybe_preload():
    if os.environ.get("STANDIN_PRELOAD", "0") == "1":
        print("[Stand-In] Preloading pipelines (STANDIN_PRELOAD=1)...")
        _base_pipe()
        try:
            _face_swap_pipe()
        except Exception as e:
            print(f"[Stand-In] Face swap preload skipped: {e}")
        try:
            _vace_pipe()
        except Exception as e:
            print(f"[Stand-In] VACE preload skipped: {e}")
        print("[Stand-In] Preloading complete.")


_maybe_preload()

with gr.Blocks(theme=gr.themes.Soft(), css="footer {display:none !important}") as demo:
    gr.Markdown("# Stand-In WebUI")
    with gr.Tab("Base IP2V"):
        with gr.Row():
            with gr.Column():
                base_ip = gr.Image(label="Face Image", type="pil", image_mode="RGB")
                base_prompt = gr.Textbox(label="Prompt", value="A man sits comfortably at a desk, facing the camera ...", lines=4)
                base_neg = gr.Textbox(label="Negative Prompt", value=NEGATIVE_DEFAULT, lines=3)
                base_seed = gr.Slider(0, 100000, value=0, step=1, label="Seed")
                base_steps = gr.Slider(10, 50, value=20, step=1, label="Steps")
                base_fps = gr.Slider(10, 30, value=16, step=1, label="FPS")
                base_quality = gr.Slider(1, 10, value=9, step=1, label="Quality")
                base_btn = gr.Button("Run Base")
            with gr.Column():
                base_out = gr.Video(label="Result", height=480)
        base_btn.click(ui_base, [base_ip, base_prompt, base_neg, base_seed, base_steps, base_fps, base_quality], base_out)
        gr.Examples(
            examples=[["test/input/lecun.jpg", "A man sits comfortably at a desk, facing the camera...", NEGATIVE_DEFAULT, 0, 20, 16, 9]],
            inputs=[base_ip, base_prompt, base_neg, base_seed, base_steps, base_fps, base_quality],
            label="Examples"
        )

    with gr.Tab("LoRA Stylization"):
        with gr.Row():
            with gr.Column():
                lora_ip = gr.Image(label="Face Image", type="pil")
                lora_prompt = gr.Textbox(label="Prompt", value="GoldenBoyStyle, a man speaking to camera", lines=4)
                lora_neg = gr.Textbox(label="Negative Prompt", value=NEGATIVE_DEFAULT, lines=3)
                lora_seed = gr.Slider(0, 100000, value=0, step=1, label="Seed")
                lora_steps = gr.Slider(10, 50, value=20, step=1, label="Steps")
                lora_path = gr.Textbox(label="LoRA Path", value="checkpoints/lora/ghibli.safetensors")
                lora_scale = gr.Slider(0.1, 3.0, value=1.5, step=0.1, label="LoRA Scale")
                lora_fps = gr.Slider(10, 30, value=16, step=1, label="FPS")
                lora_quality = gr.Slider(1, 10, value=9, step=1, label="Quality")
                lora_btn = gr.Button("Run LoRA")
            with gr.Column():
                lora_out = gr.Video(label="Result", height=480)
        lora_btn.click(ui_lora, [lora_ip, lora_prompt, lora_neg, lora_seed, lora_steps, lora_path, lora_scale, lora_fps, lora_quality], lora_out)
        gr.Examples(
            examples=[["test/input/lecun.jpg", "GoldenBoyStyle, a man speaking to camera", NEGATIVE_DEFAULT, 0, 20, "checkpoints/lora/ghibli.safetensors", 1.5, 16, 9]],
            inputs=[lora_ip, lora_prompt, lora_neg, lora_seed, lora_steps, lora_path, lora_scale, lora_fps, lora_quality],
            label="Examples"
        )

    with gr.Tab("Face Swap"):
        with gr.Row():
            with gr.Column():
                fs_ip = gr.Image(label="Source Face", type="pil")
                fs_video = gr.Textbox(label="Input Video Path", value="test/input/woman.mp4")
                fs_video_preview = gr.Video(label="Input Video Preview", value="test/input/woman.mp4", height=280)
                fs_prompt = gr.Textbox(label="Prompt", value="The video features a woman standing in front ...", lines=5)
                fs_neg = gr.Textbox(label="Negative Prompt", value=NEGATIVE_DEFAULT, lines=3)
                fs_seed = gr.Slider(0, 100000, value=0, step=1, label="Seed")
                fs_steps = gr.Slider(10, 50, value=20, step=1, label="Steps")
                fs_denoise = gr.Slider(0.1, 1.0, value=0.85, step=0.01, label="Denoising Strength")
                fs_force_bg = gr.Checkbox(label="Force Background Consistency", value=False)
                fs_fps = gr.Slider(10, 30, value=16, step=1, label="FPS")
                fs_quality = gr.Slider(1, 10, value=9, step=1, label="Quality")
                fs_btn = gr.Button("Run Face Swap")
            with gr.Column():
                fs_out = gr.Video(label="Result", height=480)
        fs_btn.click(ui_face_swap, [fs_ip, fs_video, fs_prompt, fs_neg, fs_seed, fs_steps, fs_denoise, fs_force_bg, fs_fps, fs_quality], fs_out)
        fs_video.change(_preview_video, inputs=fs_video, outputs=fs_video_preview)
        gr.Examples(
            examples=[["test/input/ruonan.jpg", "test/input/woman.mp4", "The video features a woman standing in front ...", NEGATIVE_DEFAULT, 0, 20, 0.85, False, 16, 9]],
            inputs=[fs_ip, fs_video, fs_prompt, fs_neg, fs_seed, fs_steps, fs_denoise, fs_force_bg, fs_fps, fs_quality],
            label="Examples"
        )

    with gr.Tab("VACE Motion"):
        with gr.Row():
            with gr.Column():
                v_ip = gr.Image(label="Face Image", type="pil")
                v_ref_video = gr.Textbox(label="Reference Video Path", value="test/input/pose.mp4")
                v_ref_preview = gr.Video(label="Reference Video Preview", value="test/input/pose.mp4", height=280)
                v_ref_image = gr.Image(label="Reference Image", type="pil")
                v_prompt = gr.Textbox(label="Prompt", value="A woman raises her hands.", lines=3)
                v_neg = gr.Textbox(label="Negative Prompt", value=NEGATIVE_DEFAULT, lines=3)
                v_seed = gr.Slider(0, 100000, value=0, step=1, label="Seed")
                v_steps = gr.Slider(10, 50, value=20, step=1, label="Steps")
                v_scale = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="VACE Scale")
                v_fps = gr.Slider(10, 30, value=16, step=1, label="FPS")
                v_quality = gr.Slider(1, 10, value=9, step=1, label="Quality")
                v_btn = gr.Button("Run VACE")
            with gr.Column():
                v_out = gr.Video(label="Result", height=480)
        v_btn.click(ui_vace, [v_ip, v_ref_video, v_ref_image, v_prompt, v_neg, v_seed, v_steps, v_scale, v_fps, v_quality], v_out)
        v_ref_video.change(_preview_video, inputs=v_ref_video, outputs=v_ref_preview)
        gr.Examples(
            examples=[["test/input/first_frame.png", "test/input/pose.mp4", "test/input/first_frame.png", "A woman raises her hands.", NEGATIVE_DEFAULT, 0, 20, 0.8, 16, 9]],
            inputs=[v_ip, v_ref_video, v_ref_image, v_prompt, v_neg, v_seed, v_steps, v_scale, v_fps, v_quality],
            label="Examples"
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=bool(int(os.environ.get("STANDIN_SHARE", "0"))))
