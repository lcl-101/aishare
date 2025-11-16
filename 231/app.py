"""ChronoEdit Gradio WebUI."""

import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPVisionModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
from chronoedit_diffusers.transformer_chronoedit import ChronoEditTransformer3DModel
from scripts.run_inference_diffusers import calculate_dimensions


_PIPELINE_CACHE: Dict[Tuple[str, str, Optional[str], float], ChronoEditPipeline] = {}
_PROMPT_ENHANCER_CACHE: Dict[str, Tuple[object, object]] = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()

DEFAULT_NUM_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_FLOW_SHIFT = 5.0
PROMPT_MAX_RESOLUTION = 1080


def _select_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise gr.Error("当前环境不可用 CUDA, 请切换到 CPU 设备。")
    return requested

def _pick_prompt_attn_impl(prefer_flash: bool = True) -> str:
    if prefer_flash:
        try:
            import flash_attn  # noqa: F401

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                if (major, minor) >= (8, 0):
                    return "flash_attention_2"
        except Exception:
            pass

    try:
        if torch.backends.cuda.sdp_kernel.is_available():
            return "sdpa"
    except Exception:
        pass

    return "eager"


def _load_prompt_enhancer_from_id(model_id: str, model_cls):
    attn_impl = _pick_prompt_attn_impl(True)
    load_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": attn_impl,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "cuda:0"

    model = model_cls.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _load_prompt_enhancer_model(model_name: str):
    print(f"Loading prompt enhancer model: {model_name}")

    if model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        return _load_prompt_enhancer_from_id(
            model_name,
            Qwen2_5_VLForConditionalGeneration,
        )

    if model_name == "Qwen/Qwen3-VL-30B-A3B-Instruct":
        return _load_prompt_enhancer_from_id(
            model_name,
            Qwen3VLMoeForConditionalGeneration,
        )

    model_path = Path(model_name)
    if model_path.exists():
        last_error: Optional[Exception] = None
        for candidate_cls in (
            Qwen3VLMoeForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
        ):
            try:
                return _load_prompt_enhancer_from_id(str(model_path), candidate_cls)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise ValueError(
            f"Failed to load local prompt enhancer from '{model_name}': {last_error}"
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _resize_prompt_image_if_needed(image: Image.Image, max_resolution: int = PROMPT_MAX_RESOLUTION) -> Image.Image:
    width, height = image.size
    if min(width, height) <= max_resolution:
        return image
    scale = max_resolution / float(min(width, height))
    new_size = (int(width * scale), int(height * scale))
    print(f"Resizing image for prompt enhancer from {image.size} to {new_size}")
    return image.resize(new_size, Image.LANCZOS)


def _run_prompt_enhancer(messages, model, processor) -> str:
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device).to(model.dtype)
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    elif isinstance(model, Qwen3VLMoeForConditionalGeneration):
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device).to(model.dtype)
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    else:
        raise ValueError("Unsupported prompt enhancer model type")

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0]


def _enhance_prompt_with_cot(
    image_path: str,
    input_prompt: str,
    model,
    processor,
    max_resolution: int = PROMPT_MAX_RESOLUTION,
) -> str:
    print(f"Loading image for prompt enhancement: {image_path}")
    input_image = Image.open(image_path).convert("RGB")
    input_image = _resize_prompt_image_if_needed(input_image, max_resolution)

    cot_prompt = f"""You are a professional edit instruction rewriter and prompt engineer. Your task is to generate a precise, concise, and visually achievable chain-of-thought reasoning based on the user-provided instruction and the image to be edited.

You have the following information:
1. The user provides an image (the original image to be edited)
2. question text: {input_prompt}

Your task is NOT to output the final answer or the edited image. Instead, you must:
- Generate a "thinking" or chain-of-thought process that explains how you reason about the editing task.
- First identify the task type, then provide reasoning/analysis that leads to how the image should be edited.
- Always describe pose and appearance in detail.
- Match the original visual style or genre (anime, CG art, cinematic, poster). If not explicit, choose a stylistically appropriate one based on the image.
- Incorporate motion and camera direction when relevant (e.g., walking, turning, dolly in/out, pan), implying natural human/character motion and interactions.
- Maintain quoted phrases or titles exactly (e.g., character names, series names). Do not translate or alter the original language of text.

## Task Type Handling Rules:

**1. Standard Editing Tasks (e.g., Add, Delete, Replace, Action Change):**
- For replacement tasks, specify what to replace and key visual features of the new element.
- For text editing tasks, specify text position, color, and layout concisely.
- If the user wants to "extract" something, this means they want to remove the background and only keep the specified object isolated. We should add "while removing the background" to the reasoning.
- Explicitly note what must stay unchanged: appearances (hairstyle, clothing, expression, skin tone/race, age), posture, pose, visual style/genre, spatial layout, and shot composition (e.g., medium shot, close-up, side view).

**2. Character Consistency Editing Tasks (e.g., Scenario Change):**
- For tasks that place an object/character (e.g., human, robot, animal) in a completely new scenario, preserve the object's core identity (appearance, materials, key features) but adapt its pose, interaction, and context to fit naturally in the new environment.
- Reason about how the object should interact with the new scenario (e.g., pose changes, hand positions, orientation, facial direction).
- The background and context should transform completely to match the new scenario while maintaining visual coherence.
- Describe both what stays the same (core appearance) and what must change (pose, interaction, setting) to make the scene look realistic and natural.

The length of outputs should be **around 80 - 100 words** to fully describe the transformation. Always start with "The user wants to ..."

Example Output 1 (Standard Editing Task):
The user wants to make the knight kneel on his right knee while keeping the rest of the pose intact.
The knight should lower his stance so his right leg bends to the ground in a kneeling position, with the left leg bent upright to support balance.
The shield with the NVIDIA logo should still be held up firmly in his left hand, angled forward in a defensive posture, while the right hand continues gripping the weapon.
The armor reflections, proportions, and medieval style should remain consistent, emphasizing a powerful and respectful kneeling stance.

Example Output 2 (Character Consistency Editing Task):
The user wants to change the image by modifying the scene so that the woman is drinking coffee in a cozy coffee shop.
The elegant anime-style woman keeps her same graceful expression, long flowing dark hair adorned with golden ornaments, and detailed traditional outfit with red and gold floral patterns.
She is now seated at a wooden café table, holding a steaming cup of coffee near her lips with one hand, while soft sunlight filters through the window, highlighting her refined features.
The background transforms into a warmly lit café interior with subtle reflections, bookshelves, and gentle ambience, maintaining the delicate, painterly aesthetic.
"""

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": cot_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": input_image},
            ],
        },
    ]

    print("Generating enhanced prompt with chain-of-thought reasoning...")
    return _run_prompt_enhancer(messages, model, processor)


def _load_pipeline(
    model_path: str,
    device: str,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
) -> ChronoEditPipeline:
    cache_key = (
        model_path,
        device,
        lora_path if lora_path else None,
        float(lora_scale) if lora_path else 0.0,
    )

    with _CACHE_LOCK:
        pipe = _PIPELINE_CACHE.get(cache_key)
        if pipe is not None:
            pipe.to(device)
            return pipe

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    image_encoder = CLIPVisionModel.from_pretrained(
        model_path,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
    )
    transformer = ChronoEditTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = ChronoEditPipeline.from_pretrained(
        model_path,
        image_encoder=image_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=dtype,
    )

    if lora_path:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)

    pipe.to(device)

    with _CACHE_LOCK:
        _PIPELINE_CACHE[cache_key] = pipe

    return pipe

def _prepare_prompt(
    image_path: str,
    prompt: str,
    use_prompt_enhancer: bool,
    enhancer_model_name: str,
) -> Tuple[str, Optional[str]]:
    if not use_prompt_enhancer:
        return prompt, None

    with _CACHE_LOCK:
        cached = _PROMPT_ENHANCER_CACHE.get(enhancer_model_name)
        if cached is None:
            prompt_model, processor = _load_prompt_enhancer_model(enhancer_model_name)
            _PROMPT_ENHANCER_CACHE[enhancer_model_name] = (prompt_model, processor)
        else:
            prompt_model, processor = cached

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_model.to(target_device)

    enhanced = _enhance_prompt_with_cot(
        image_path=image_path,
        input_prompt=prompt,
        model=prompt_model,
        processor=processor,
    )

    prompt_model.to("cpu")
    torch.cuda.empty_cache()
    return enhanced, enhanced


def _render_outputs(frames, temp_dir: Path) -> Tuple[str, str]:
    video_path = temp_dir / "output.mp4"
    export_to_video(frames, str(video_path), fps=8)

    last_frame = (frames[-1] * 255).clip(0, 255).astype("uint8")
    image_path = temp_dir / "output.png"
    Image.fromarray(last_frame).save(image_path)
    return str(video_path), str(image_path)


def run_inference(
    image_path: str,
    prompt: str,
    negative_prompt: str,
    use_prompt_enhancer: bool,
    enhancer_model_name: str,
    offload_model: bool,
    model_path: str,
    seed: Optional[float],
    enable_temporal: bool,
    num_temporal_steps: int,
    device: str,
) -> Tuple[str, str, str]:
    if not image_path:
        raise gr.Error("请先上传或选择输入图像。")

    model_path = model_path.strip()
    if not model_path:
        raise gr.Error("模型路径不能为空。")

    if not Path(model_path).exists():
        raise gr.Error(f"模型路径不存在: {model_path}")

    device = _select_device(device)

    final_prompt, enhanced_prompt_text = _prepare_prompt(
        image_path,
        prompt,
        use_prompt_enhancer,
        enhancer_model_name,
    )
    if enhanced_prompt_text is None:
        enhanced_prompt_text = f"Prompt enhancer disabled. Using original prompt:\n{prompt}"

    pipe = _load_pipeline(model_path, device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=DEFAULT_FLOW_SHIFT,
    )

    image = load_image(image_path)
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    width, height = calculate_dimensions(image, mod_value)
    image = image.resize((width, height))

    generator = None
    if seed is not None and str(seed).strip():
        generator_device = device if device != "cuda" else "cuda"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    num_frames = 29 if enable_temporal else 5

    with _INFERENCE_LOCK:
        with torch.inference_mode():
            result = pipe(
                image=image,
                prompt=final_prompt,
                negative_prompt=negative_prompt.strip() or None,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=DEFAULT_NUM_STEPS,
                guidance_scale=DEFAULT_GUIDANCE_SCALE,
                enable_temporal_reasoning=enable_temporal,
                num_temporal_reasoning_steps=num_temporal_steps,
                generator=generator,
                offload_model=offload_model,
            )
    frames = result.frames[0]

    temp_dir = Path(tempfile.mkdtemp(prefix="chronoedit_ui_"))
    video_path, image_output_path = _render_outputs(frames, temp_dir)

    if device == "cuda":
        torch.cuda.empty_cache()

    return enhanced_prompt_text, video_path, image_output_path


def run_super_resolution(
    image_path: str,
    prompt: str,
    target_width: int,
    target_height: int,
    lora_scale: float,
    offload_model: bool,
    model_path: str,
    lora_path: str,
    seed: Optional[float],
    device: str,
) -> Tuple[str, str, Tuple[np.ndarray, np.ndarray]]:
    if not image_path:
        raise gr.Error("请先上传或选择输入图像。")

    model_path = model_path.strip()
    if not model_path:
        raise gr.Error("模型路径不能为空。")

    if not Path(model_path).exists():
        raise gr.Error(f"模型路径不存在: {model_path}")
    
    lora_path = lora_path.strip()
    if not lora_path:
        raise gr.Error("LoRA 路径不能为空。")
    
    if not Path(lora_path).exists():
        raise gr.Error(f"LoRA 路径不存在: {lora_path}")

    device = _select_device(device)

    pipe = _load_pipeline(
        model_path=model_path,
        device=device,
        lora_path=lora_path,
        lora_scale=lora_scale,
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=DEFAULT_FLOW_SHIFT,
    )

    original_image = load_image(image_path).convert("RGB")
    resized_input = original_image.resize((target_width, target_height))
    # Keep a copy for comparison slider before inference modifies tensors
    compare_before = resized_input.copy()
    image = resized_input

    generator = None
    if seed is not None and str(seed).strip():
        generator_device = device if device != "cuda" else "cuda"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    # Super resolution uses 5 frames (no temporal reasoning)
    num_frames = 5

    with _INFERENCE_LOCK:
        with torch.inference_mode():
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=None,
                height=target_height,
                width=target_width,
                num_frames=num_frames,
                num_inference_steps=DEFAULT_NUM_STEPS,
                guidance_scale=DEFAULT_GUIDANCE_SCALE,
                enable_temporal_reasoning=False,
                num_temporal_reasoning_steps=0,
                generator=generator,
                offload_model=offload_model,
            )
    frames = result.frames[0]

    temp_dir = Path(tempfile.mkdtemp(prefix="chronoedit_sr_"))
    video_path, image_output_path = _render_outputs(frames, temp_dir)

    with Image.open(image_output_path) as sr_image_file:
        sr_image = sr_image_file.convert("RGB")

    if sr_image.size != compare_before.size:
        sr_image = sr_image.resize(compare_before.size, Image.LANCZOS)

    comparison_slider_value = (
        np.asarray(compare_before, dtype=np.uint8),
        np.asarray(sr_image, dtype=np.uint8),
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    return video_path, image_output_path, comparison_slider_value


def build_interface() -> gr.Blocks:
    device_default = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(title="ChronoEdit Gradio UI") as demo:
        gr.Markdown("# ChronoEdit Diffusers WebUI\n使用下方控件上传图像并生成视频编辑结果。")

        with gr.Tabs():
            # Tab 1: Image Editing
            with gr.Tab("图像编辑"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="输入图像",
                            type="filepath",
                        )
                        prompt_input = gr.Textbox(
                            label="编辑提示词",
                            value="Add a sunglasses to the cat's face",
                        )
                        negative_prompt_input = gr.Textbox(
                            label="负面提示词",
                            value="",
                        )
                        use_prompt_enhancer_input = gr.Checkbox(
                            label="启用 Prompt Enhancer",
                            value=True,
                        )
                        enhancer_model_input = gr.Dropdown(
                            label="Prompt Enhancer 模型",
                            choices=[
                                "Qwen/Qwen2.5-VL-7B-Instruct",
                                "Qwen/Qwen3-VL-30B-A3B-Instruct",
                                "checkpoints/Qwen3-VL-30B-A3B-Instruct",
                            ],
                            value="checkpoints/Qwen3-VL-30B-A3B-Instruct",
                        )
                        offload_model_input = gr.Checkbox(
                            label="推理时启用模型重载 (offload)",
                            value=True,
                        )
                        model_path_input = gr.Textbox(
                            label="Diffusers 模型路径",
                            value="checkpoints/ChronoEdit-14B-Diffusers",
                        )
                        seed_input = gr.Number(
                            label="随机种子 (可选)",
                            value=None,
                        )
                        enable_temporal_input = gr.Checkbox(
                            label="启用时序推理",
                            value=False,
                        )
                        num_temporal_steps_input = gr.Slider(
                            label="时序推理步数",
                            minimum=1,
                            maximum=150,
                            value=50,
                            step=1,
                        )
                        device_input = gr.Radio(
                            label="推理设备",
                            choices=["cuda", "cpu"],
                            value=device_default,
                        )
                        run_button = gr.Button("开始推理")

                    with gr.Column():
                        enhanced_prompt_output = gr.Textbox(
                            label="增强提示词",
                            lines=6,
                        )
                        video_output = gr.Video(
                            label="生成视频",
                            interactive=False,
                        )
                        image_output = gr.Image(
                            label="最终图像",
                            type="filepath",
                        )

                examples = [
                    [
                        "assets/images/input_2.png",
                        "Add a sunglasses to the cat's face",
                        "",
                        True,
                        "checkpoints/Qwen3-VL-30B-A3B-Instruct",
                        True,
                        "checkpoints/ChronoEdit-14B-Diffusers",
                        None,
                        False,
                        50,
                        device_default,
                    ],
                ]

                gr.Examples(
                    examples=examples,
                    inputs=[
                        image_input,
                        prompt_input,
                        negative_prompt_input,
                        use_prompt_enhancer_input,
                        enhancer_model_input,
                        offload_model_input,
                        model_path_input,
                        seed_input,
                        enable_temporal_input,
                        num_temporal_steps_input,
                        device_input,
                    ],
                    label="示例",
                )

                run_button.click(
                    fn=run_inference,
                    inputs=[
                        image_input,
                        prompt_input,
                        negative_prompt_input,
                        use_prompt_enhancer_input,
                        enhancer_model_input,
                        offload_model_input,
                        model_path_input,
                        seed_input,
                        enable_temporal_input,
                        num_temporal_steps_input,
                        device_input,
                    ],
                    outputs=[
                        enhanced_prompt_output,
                        video_output,
                        image_output,
                    ],
                )

            # Tab 2: Super Resolution
            with gr.Tab("超分"):
                gr.Markdown("### 图像超分辨率增强\n使用 Upscaler LoRA 提升图像分辨率和清晰度。")
                
                with gr.Row():
                    with gr.Column():
                        sr_image_input = gr.Image(
                            label="输入低分辨率图像",
                            type="filepath",
                        )
                        sr_prompt_input = gr.Textbox(
                            label="超分提示词",
                            value="The user want to enhance image clarity and resolution while keeping the content identical. super-resolution, high detail, 4K clarity, same composition, natural texture.",
                            lines=3,
                        )
                        
                        with gr.Row():
                            sr_width_input = gr.Number(
                                label="目标宽度",
                                value=1584,
                                precision=0,
                            )
                            sr_height_input = gr.Number(
                                label="目标高度",
                                value=1056,
                                precision=0,
                            )
                        
                        sr_lora_scale_input = gr.Slider(
                            label="LoRA 强度",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                        )
                        sr_offload_model_input = gr.Checkbox(
                            label="推理时启用模型重载 (offload)",
                            value=True,
                        )
                        sr_model_path_input = gr.Textbox(
                            label="Diffusers 模型路径",
                            value="checkpoints/ChronoEdit-14B-Diffusers",
                        )
                        sr_lora_path_input = gr.Textbox(
                            label="Upscaler LoRA 路径",
                            value="checkpoints/ChronoEdit-14B-Diffusers-Upscaler-Lora/upsample_lora_diffusers.safetensors",
                        )
                        sr_seed_input = gr.Number(
                            label="随机种子 (可选)",
                            value=42,
                        )
                        sr_device_input = gr.Radio(
                            label="推理设备",
                            choices=["cuda", "cpu"],
                            value=device_default,
                        )
                        sr_run_button = gr.Button("开始超分", variant="primary")

                    with gr.Column():
                        sr_video_output = gr.Video(
                            label="处理过程视频",
                            interactive=False,
                        )
                        sr_image_output = gr.Image(
                            label="超分结果图像",
                            type="filepath",
                        )
                        sr_compare_output = gr.ImageSlider(
                            label="原图 vs 超分对比",
                            interactive=False,
                            show_download_button=False,
                            show_fullscreen_button=True,
                            max_height=512,
                            type="numpy",
                        )

                sr_examples = [
                    [
                        "assets/images/lr.png",
                        "The user want to enhance image clarity and resolution while keeping the content identical. super-resolution, high detail, 4K clarity, same composition, natural texture.",
                        1584,
                        1056,
                        1.0,
                        True,
                        "checkpoints/ChronoEdit-14B-Diffusers",
                        "checkpoints/ChronoEdit-14B-Diffusers-Upscaler-Lora/upsample_lora_diffusers.safetensors",
                        42,
                        device_default,
                    ],
                ]

                gr.Examples(
                    examples=sr_examples,
                    inputs=[
                        sr_image_input,
                        sr_prompt_input,
                        sr_width_input,
                        sr_height_input,
                        sr_lora_scale_input,
                        sr_offload_model_input,
                        sr_model_path_input,
                        sr_lora_path_input,
                        sr_seed_input,
                        sr_device_input,
                    ],
                    label="示例",
                )

                sr_run_button.click(
                    fn=run_super_resolution,
                    inputs=[
                        sr_image_input,
                        sr_prompt_input,
                        sr_width_input,
                        sr_height_input,
                        sr_lora_scale_input,
                        sr_offload_model_input,
                        sr_model_path_input,
                        sr_lora_path_input,
                        sr_seed_input,
                        sr_device_input,
                    ],
                    outputs=[
                        sr_video_output,
                        sr_image_output,
                        sr_compare_output,
                    ],
                )

    return demo


def main():
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
