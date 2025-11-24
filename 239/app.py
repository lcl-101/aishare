#!/usr/bin/env python3
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.

import os
import tempfile
import threading
from argparse import Namespace
from functools import wraps
from pathlib import Path

import gradio as gr
import imageio
from PIL import Image
import torch
import torch.distributed as dist

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_PIPELINE_CACHE = {}
_PIPELINE_LOCK = threading.Lock()
_RUNTIME_READY = False
_SR_SEED_PATCHED = False

DEFAULT_PROMPT = "A cinematic close-up of a fox in the snow, film lighting"
DEFAULT_UI_CONFIG = {
    "model_path": "./checkpoints/HunyuanVideo-1.5",
    "resolution": "480p",
    "aspect_ratio": "16:9",
    "video_length": 121,
    "steps": 50,
    "seed": 123,
    "enable_sr": False,
    "cfg_distilled": False,
    "dtype": "bf16",
    "enable_offloading": False,
}

EXAMPLE_PROMPTS = [
    "俯视角度，一位有着深色，略带凌乱的长卷发的年轻中国女性，佩戴着闪耀的珍珠项链和圆形金色耳环，她凌乱的头发被风吹散，她微微抬头，望向天空，神情十分哀伤，眼中含着泪水。嘴唇涂着红色口红。背景是带有华丽红色花纹的图案。画面呈现复古电影风格，色调低饱和，带着轻微柔焦，烘托情绪氛围，质感仿佛20世纪90年代的经典胶片风格，营造出怀旧且富有戏剧性的感觉。",
    "slowly advancing medium shot, shot from a level angle, focuses on the center of an empty football field, where a DJ is immersed in his musical world. He wears a pair of professional, matte-black headphones, one earcup slightly removed, revealing a focused expression and a brow beaded with sweat from his intense concentration. He wears a black bomber jacket, zipped open to reveal a T-shirt underneath. His upper body sways back and forth rhythmically to the throbbing electronic beats, his head moving with precise movement. The mixing console in front of him serves as the primary source of light. In the distance, the cool white glow of several stadium floodlights casts a deep, dark haze across the vast field, casting long shadows across the emerald green grass, creating a stark contrast to the brightly lit area surrounding the DJ booth. His hands danced swiftly and precisely across the equipment, one hand steadily pushing and pulling a long volume fader, while the fingers of the other nimbly jumped between the illuminated knobs and pads, sometimes decisively cutting a bass line, sometimes triggering an echo effect. The entire scene was filled with high-tech dynamics and the solitary creative passion. Against the backdrop of the vast and silent night stadium, it created an atmosphere of high focus, energy, and a slightly surreal feeling.",
    "Handheld lens shooting, the camera focuses on the wall clock hanging on the green-toned wall, shaking slightly. The second hand sweeps steadily across the clock face, and the shadow of the clock cast on the wall shifts subtly with the movement of the lens.",
    "On a wet street corner in a cyberpunk city at night, a large neon sign reading \"Hunyuan Video 1.5\" lights up sequentially, illuminating the dark, rainy environment with a pinkish-purple glow. The scene is a dark, rain-slicked street corner in a futuristic, cinematic cyberpunk city. Mounted on the metallic, weathered facade of a building is a massive, unlit neon sign. The sign's glass tube framework clearly spells out the words \"Hunyuan Video 1.5\". Initially, the street is dimly lit, with ambient light from distant skyscrapers creating shimmering reflections on the wet asphalt below. Then, the camera zooms in slowly toward the sign. As it moves, a low electrical sizzling sound begins. In the background, the dense urban landscape of the cyberpunk metropolis is visible through a light atmospheric haze, with towering structures adorned with their own flickering advertisements. A complex web of cables and pipes crisscrosses between the buildings. The shot is at a low angle, looking up at the sign to emphasize its grand scale. The lighting is high-contrast and dramatic, dominated by the neon glow which creates sharp, specular reflections and deep shadows. The atmosphere is moody and tech-noir. The overall video presents a cinematic photography realistic style.",
    "In a sleek museum gallery, a woman receives a glass of wine poured directly from an animated oil painting. A sophisticated woman with dark hair tied back elegantly stands in the mid-ground. She is wearing a simple, black silk sleeveless dress and holds a clear, crystal wine glass in her right hand. She is positioned before a large, baroque-style oil painting in an ornate, gilded frame. Inside the painting, an aristocratic man with a mustache, dressed in a dark velvet doublet with a white lace collar, is depicted. His form is defined by visible, impasto oil brushstrokes. Initially, the woman watches the painting with calm poise. Then, the painted man's arm slowly animates, his painted texture retained as he lifts a dark bottle. Next, a photorealistic stream of red wine emerges directly from the flat canvas surface, arcing through the air and splashing gently into the real crystal glass she holds. She remains perfectly still, accepting the impossible pour with a subtle, knowing smile. The setting is a modern art gallery with high white walls and polished dark concrete floors that reflect the ambient light. Focused track lighting from the high ceiling casts a warm, dramatic spotlight on the woman and the painting, creating soft shadows. In the background, two other gallery patrons, a man and a woman in stylish, modern attire, stroll slowly from right to left, their figures slightly blurred by a shallow depth of field, moving naturally through the hall. The shot is at an eye-level angle with the woman. The camera remains static, capturing the surreal event in a steady medium shot. The lighting is high-contrast and dramatic, reminiscent of a cinematic photography realistic style, using soft side lighting to accentuate the woman's features and the texture of the painting. The mood is surreal, elegant, and mysterious. The overall video presents a cinematic photography realistic style.",
]


def _init_process_group():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    except Exception:
        if backend != "gloo":
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        else:
            raise


def _ensure_runtime_ready():
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return
    _init_process_group()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
    initialize_infer_state(
        Namespace(
            sage_blocks_range="0-53",
            use_sageattn=False,
            enable_torch_compile=False,
        )
    )
    _RUNTIME_READY = True


def _build_transformer_version(resolution: str, task: str, cfg_distilled: bool) -> str:
    version = f"{resolution}_{task}"
    if cfg_distilled:
        version += "_distilled"
    return version


def _preload_default_pipeline():
    model_path = DEFAULT_UI_CONFIG["model_path"]
    if not os.path.isdir(model_path):
        print(f"[Preload] 模型目录 {model_path} 不存在，跳过预加载。")
        return
    try:
        transformer_version = _build_transformer_version(
            DEFAULT_UI_CONFIG["resolution"],
            "t2v",
            DEFAULT_UI_CONFIG["cfg_distilled"],
        )
        _load_pipeline(
            model_path=model_path,
            transformer_version=transformer_version,
            enable_sr=DEFAULT_UI_CONFIG["enable_sr"],
            dtype=DEFAULT_UI_CONFIG["dtype"],
            enable_offloading=DEFAULT_UI_CONFIG["enable_offloading"],
        )
        print(f"[Preload] 已加载 {transformer_version} 到缓存。")
    except Exception as exc:
        print(f"[Preload] 预加载失败：{exc}")


def _load_pipeline(
    model_path: str,
    transformer_version: str,
    enable_sr: bool,
    dtype: str,
    enable_offloading: bool,
) -> HunyuanVideo_1_5_Pipeline:
    _ensure_runtime_ready()
    _ensure_sr_seed_patch()
    dtype_obj = _DTYPES[dtype]
    cache_key = (
        Path(model_path).resolve(),
        transformer_version,
        enable_sr,
        dtype_obj,
        enable_offloading,
    )
    with _PIPELINE_LOCK:
        pipe = _PIPELINE_CACHE.get(cache_key)
        if pipe is not None:
            return pipe
        pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=model_path,
            transformer_version=transformer_version,
            transformer_dtype=dtype_obj,
            create_sr_pipeline=enable_sr,
            enable_offloading=enable_offloading,
            enable_group_offloading=False,
            overlap_group_offloading=True,
            force_sparse_attn=False,
        )
        _PIPELINE_CACHE[cache_key] = pipe
        return pipe


def _ensure_sr_seed_patch() -> None:
    global _SR_SEED_PATCHED
    if _SR_SEED_PATCHED:
        return

    try:
        from hyvideo.pipelines.hunyuan_video_sr_pipeline import HunyuanVideo_1_5_SR_Pipeline
    except ImportError:
        return

    original_call = HunyuanVideo_1_5_SR_Pipeline.__call__

    @wraps(original_call)
    def patched_call(self, *args, **kwargs):
        generator = kwargs.get("generator")
        seed = kwargs.get("seed")

        if generator is not None:
            try:
                state = generator.get_state()
            except Exception:
                state = None
            generator = torch.Generator(device=torch.device("cpu"))
            if state is not None:
                generator.set_state(state)
            kwargs["generator"] = generator
        elif seed is not None:
            try:
                seed_int = int(seed)
            except (TypeError, ValueError):
                seed_int = None
            if seed_int is not None and seed_int >= 0:
                kwargs["generator"] = torch.Generator(device=torch.device("cpu")).manual_seed(seed_int)

        return original_call(self, *args, **kwargs)

    HunyuanVideo_1_5_SR_Pipeline.__call__ = patched_call
    _SR_SEED_PATCHED = True


def _tensor_to_video_tensor(video_tensor: torch.Tensor) -> torch.Tensor:
    if video_tensor.ndim == 5:
        return video_tensor[0]
    return video_tensor


def _tensor_to_frame(video_tensor: torch.Tensor) -> Image.Image:
    tensor = _tensor_to_video_tensor(video_tensor).detach().cpu()
    frame = tensor[:, 0, :, :]
    frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
    frame = frame.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(frame)


def _tensor_to_video_file(video_tensor: torch.Tensor) -> str:
    tensor = _tensor_to_video_tensor(video_tensor).detach().cpu()
    frames = (tensor.clamp(0, 1) * 255).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).contiguous().numpy()
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    imageio.mimwrite(handle.name, frames, fps=24)
    handle.close()
    return handle.name


def generate_media(
    prompt: str,
    negative_prompt: str,
    aspect_ratio: str,
    resolution: str,
    video_length: int,
    steps: int,
    seed: int,
    enable_sr: bool,
    enable_rewrite: bool,
    model_path: str,
    dtype: str,
    enable_offloading: bool,
    cfg_distilled: bool,
    reference_image,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required.")
    model_path = model_path.strip() or "ckpts"
    if not os.path.isdir(model_path):
        raise gr.Error(f"Model path '{model_path}' does not exist.")
    if dtype not in _DTYPES:
        raise gr.Error(f"Unsupported dtype: {dtype}")
    task = "i2v" if reference_image is not None else "t2v"
    transformer_version = _build_transformer_version(resolution, task, cfg_distilled)
    pipe = _load_pipeline(
        model_path=model_path,
        transformer_version=transformer_version,
        enable_sr=enable_sr,
        dtype=dtype,
        enable_offloading=enable_offloading,
    )

    seed_value = None
    if seed is not None and str(seed).strip() != "":
        seed_int = int(seed)
        if seed_int >= 0:
            seed_value = seed_int

    output = pipe(
        enable_sr=enable_sr,
        prompt=prompt.strip(),
        aspect_ratio=aspect_ratio,
        num_inference_steps=int(steps),
        video_length=int(video_length),
        negative_prompt=negative_prompt or "",
        seed=seed_value,
        prompt_rewrite=enable_rewrite,
        reference_image=reference_image,
        output_type="pt",
        return_pre_sr_video=False,
    )

    video_tensor = output.sr_videos if enable_sr and getattr(output, "sr_videos", None) is not None else output.videos
    preview = _tensor_to_frame(video_tensor)
    video_path = _tensor_to_video_file(video_tensor)

    metadata = (
        f"Resolution: {resolution}\n"
        f"Aspect ratio: {aspect_ratio}\n"
        f"Video length: {int(video_length)} frames\n"
        f"Inference steps: {int(steps)}\n"
        f"Seed: {seed_value if seed_value is not None else 'random'}\n"
        f"Transformer version: {transformer_version}\n"
        f"Video file: {video_path}"
    )

    return preview, video_path, metadata, video_path


def build_demo():
    with gr.Blocks(title="HunyuanVideo 1.5 Gradio") as demo:
        gr.Markdown(
            """
            ## HunyuanVideo-1.5 Playground
            - 单卡即可运行，默认开启 offloading 减少显存占用。
            - 勾选 `Enable Rewrite` 前请先设置 README 中的 vLLM 环境变量。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the scene...",
                    value=DEFAULT_PROMPT,
                )
                gr.Examples(
                    examples=[[text] for text in EXAMPLE_PROMPTS],
                    inputs=[prompt],
                    label="Prompt Examples",
                )
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Things to avoid", lines=2)
                reference_image = gr.Image(label="Reference Image (optional)", type="pil")
                model_path = gr.Textbox(label="Model Path", value=DEFAULT_UI_CONFIG["model_path"])
                resolution = gr.Dropdown(["480p", "720p"], value=DEFAULT_UI_CONFIG["resolution"], label="Base Resolution")
                aspect_ratio = gr.Dropdown(["16:9", "9:16", "4:3", "1:1"], value=DEFAULT_UI_CONFIG["aspect_ratio"], label="Aspect Ratio")
                video_length = gr.Slider(16, 121, value=DEFAULT_UI_CONFIG["video_length"], step=1, label="Video Length (frames)")
                steps = gr.Slider(20, 80, value=DEFAULT_UI_CONFIG["steps"], step=1, label="Denoising Steps")
                seed = gr.Number(label="Seed (>=0 for deterministic)", value=DEFAULT_UI_CONFIG["seed"])
                enable_sr = gr.Checkbox(label="Enable Super Resolution", value=DEFAULT_UI_CONFIG["enable_sr"])
                cfg_distilled = gr.Checkbox(label="Use Distilled Transformer", value=DEFAULT_UI_CONFIG["cfg_distilled"])
                dtype = gr.Radio(["bf16", "fp32"], value=DEFAULT_UI_CONFIG["dtype"], label="Transformer dtype")
                enable_rewrite = gr.Checkbox(label="Enable Rewrite", value=False)
                advanced = gr.Accordion("Memory Options", open=False)
                with advanced:
                    enable_offloading = gr.Checkbox(label="Enable Offloading", value=DEFAULT_UI_CONFIG["enable_offloading"])

                run_button = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                preview = gr.Image(label="Preview Frame")
                video = gr.Video(label="Generated Video")
                metadata = gr.Textbox(label="Run Metadata", lines=6)
                download = gr.File(label="Download MP4")

        inputs = [
            prompt,
            negative_prompt,
            aspect_ratio,
            resolution,
            video_length,
            steps,
            seed,
            enable_sr,
            enable_rewrite,
            model_path,
            dtype,
            enable_offloading,
            cfg_distilled,
            reference_image,
        ]
        outputs = [preview, video, metadata, download]
        run_button.click(generate_media, inputs=inputs, outputs=outputs)

    return demo


def main():
    _preload_default_pipeline()
    demo = build_demo()
    demo.queue().launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
