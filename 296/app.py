#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UniVideo Gradio Web 演示程序
基于 UniVideo 的多任务视频生成与编辑
"""

import os
import torch
import numpy as np
import yaml
import gradio as gr
from PIL import Image
from pathlib import Path

from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformer_univideo_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
from mllm_encoder import MLLMInContext, MLLMInContextConfig
from pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig

from utils import pad_image_pil_to_square, load_model


# 全局变量存储 pipeline
pipeline = None
current_variant = None

# 本地模型路径配置
LOCAL_MODEL_BASE = "checkpoints"
LOCAL_HUNYUAN_PATH = os.path.join(LOCAL_MODEL_BASE, "HunyuanVideo")
LOCAL_QWEN_PATH = os.path.join(LOCAL_MODEL_BASE, "Qwen2.5-VL-7B-Instruct")
LOCAL_UNIVIDEO_PATH = os.path.join(LOCAL_MODEL_BASE, "UniVideo")

# 默认负面提示词
NEGATIVE_PROMPT = "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards, computer-generated environment, weak dynamics, distorted and erratic motions, unstable framing and a disorganized composition."

# Demo 路径
DEMO_PATH = "demo"

# 配置文件路径
CONFIG_PATH = "configs"


def load_pipeline(variant="variant1"):
    """加载指定变体的 pipeline"""
    global pipeline, current_variant
    
    if pipeline is not None and current_variant == variant:
        return pipeline
    
    # 释放之前的 pipeline
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()
    
    # 选择配置文件
    if variant == "variant1":
        config_path = os.path.join(CONFIG_PATH, "univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml")
        transformer_ckpt = os.path.join(LOCAL_UNIVIDEO_PATH, "univideo_qwen2p5vl7b_hidden_hunyuanvideo/model.ckpt")
        mllm_encoder_ckpt = None
    else:
        config_path = os.path.join(CONFIG_PATH, "univideo_qwen2p5vl7b_queries_hunyuanvideo.yaml")
        transformer_ckpt = os.path.join(LOCAL_UNIVIDEO_PATH, "univideo_qwen2p5vl7b_queries_hunyuanvideo/model.ckpt")
        mllm_encoder_ckpt = os.path.join(LOCAL_UNIVIDEO_PATH, "univideo_qwen2p5vl7b_queries_hunyuanvideo/mllm.ckpt")
    
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    
    # 修改配置以使用本地路径
    raw["mllm_config"]["mllm_id"] = LOCAL_QWEN_PATH
    raw["pipeline_config"]["hunyuan_model_id"] = LOCAL_HUNYUAN_PATH
    
    mllm_config = MLLMInContextConfig(**raw["mllm_config"])
    pipe_cfg = UniVideoPipelineConfig(**raw["pipeline_config"])
    
    # 创建 MLLM encoder
    mllm_encoder = MLLMInContext(mllm_config)
    
    # 加载 mllm_encoder checkpoint (variant2 需要)
    if mllm_encoder_ckpt is not None:
        print(f"[初始化] 正在加载 mllm_encoder 检查点: {mllm_encoder_ckpt}")
        mllm_encoder = load_model(mllm_encoder, mllm_encoder_ckpt)
    mllm_encoder.requires_grad_(False)
    mllm_encoder.eval()
    
    # 加载 VAE
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        LOCAL_HUNYUAN_PATH,
        subfolder="vae",
        low_cpu_mem_usage=False,
        device_map=None
    )
    vae.eval()
    
    # 加载 transformer
    qwenvl_txt_dim = 3584
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        LOCAL_HUNYUAN_PATH,
        subfolder="transformer",
        low_cpu_mem_usage=False,
        device_map=None,
        text_embed_dim=qwenvl_txt_dim
    )
    transformer.qwen_project_in = TwoLayerMLP(qwenvl_txt_dim, qwenvl_txt_dim * 4, 4096)
    with torch.no_grad():
        torch.nn.init.ones_(transformer.qwen_project_in.ln.weight)
        for layer in transformer.qwen_project_in.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
    
    # 加载 transformer checkpoint
    def rename_func(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
            new_state_dict[new_k] = v
        return new_state_dict
    
    print(f"[初始化] 正在加载 transformer 检查点: {transformer_ckpt}")
    transformer = load_model(transformer, transformer_ckpt, rename_func=rename_func)
    
    # 加载 scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        LOCAL_HUNYUAN_PATH,
        subfolder="scheduler"
    )
    
    # 构建 pipeline
    pipeline = UniVideoPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm_encoder=mllm_encoder,
        univideo_config=pipe_cfg
    ).to(device="cuda", dtype=torch.bfloat16)
    
    current_variant = variant
    print(f"[初始化] Pipeline 加载完成，使用变体: {variant}")
    
    return pipeline


def process_output(output, output_path):
    """处理 pipeline 输出"""
    # 文本输出
    if hasattr(output, "text") and output.text is not None:
        return output.text[0] if output.text else ""
    
    # 图像/视频输出
    elif hasattr(output, "frames"):
        frames = output.frames[0]  # (F, H, W, C)
        
        if hasattr(frames, "detach"):
            frames = frames.detach().cpu().float().numpy()
        
        F, H, W, C = frames.shape
        
        # 图像输出
        if F == 1:
            img = frames[0]
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            img = (img * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(output_path)
            return output_path
        # 视频输出
        else:
            export_to_video(frames, output_path, fps=24)
            return output_path
    
    return None


# ==================== 任务处理函数 ====================

def run_understanding(variant, video_file, prompt, max_video_frames, seed):
    """视频理解任务"""
    pipe = load_pipeline(variant)
    
    if video_file is None:
        return "请上传视频文件"
    
    # 限制视频帧数以避免显存不足
    # Qwen2.5-VL处理视频时会生成大量视觉tokens，帧数过多会导致OOM
    max_frames = int(max_video_frames)
    
    output = pipe(
        prompts=[prompt],
        cond_video_path=video_file,
        num_frames=max_frames,
        height=480,  # 降低分辨率减少显存占用
        width=854,
        seed=int(seed),
        task="understanding",
    )
    
    if hasattr(output, "text") and output.text is not None:
        return output.text[0] if output.text else "无输出"
    return "无输出"


def run_t2v(variant, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed):
    """文本生成视频"""
    pipe = load_pipeline(variant)
    
    output_path = "outputs/t2v_output.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    output = pipe(
        prompts=[prompt],
        negative_prompt=NEGATIVE_PROMPT,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        image_guidance_scale=float(image_guidance_scale),
        seed=int(seed),
        timestep_shift=7.0,
        task="t2v",
    )
    
    return process_output(output, output_path)


def run_i2v(variant, image_file, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed):
    """图像生成视频"""
    pipe = load_pipeline(variant)
    
    if image_file is None:
        return None
    
    output_path = "outputs/i2v_output.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    output = pipe(
        prompts=[prompt],
        negative_prompt=NEGATIVE_PROMPT,
        cond_image_path=image_file,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        image_guidance_scale=float(image_guidance_scale),
        seed=int(seed),
        timestep_shift=7.0,
        task="i2v",
    )
    
    return process_output(output, output_path)


def run_i2i_edit(variant, image_file, prompt, height, width, num_steps, guidance_scale, image_guidance_scale, seed):
    """图像编辑"""
    pipe = load_pipeline(variant)
    
    if image_file is None:
        return None
    
    output_path = "outputs/i2i_edit_output.jpg"
    os.makedirs("outputs", exist_ok=True)
    
    output = pipe(
        prompts=[prompt],
        negative_prompt=NEGATIVE_PROMPT,
        cond_image_path=image_file,
        height=int(height),
        width=int(width),
        num_frames=1,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        image_guidance_scale=float(image_guidance_scale),
        seed=int(seed),
        timestep_shift=7.0,
        task="i2i_edit",
    )
    
    return process_output(output, output_path)


def run_v2v_edit(variant, video_file, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed):
    """视频编辑"""
    pipe = load_pipeline(variant)
    
    if video_file is None:
        return None
    
    output_path = "outputs/v2v_edit_output.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    output = pipe(
        prompts=[prompt],
        negative_prompt=NEGATIVE_PROMPT,
        cond_video_path=video_file,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        image_guidance_scale=float(image_guidance_scale),
        seed=int(seed),
        timestep_shift=7.0,
        task="v2v_edit",
    )
    
    return process_output(output, output_path)


def run_multiid(variant, ref_images, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed):
    """多身份上下文生成"""
    pipe = load_pipeline(variant)
    
    if ref_images is None or len(ref_images) == 0:
        return None
    
    output_path = "outputs/multiid_output.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    # 处理参考图像 (Gallery 返回的是 [(filepath, caption), ...] 或 [filepath, ...])
    pil_images = []
    for img in ref_images:
        if isinstance(img, tuple):
            # Gallery format: (filepath, caption)
            pil_images.append(pad_image_pil_to_square(Image.open(img[0]).convert("RGB")))
        elif isinstance(img, str):
            pil_images.append(pad_image_pil_to_square(Image.open(img).convert("RGB")))
        elif isinstance(img, Image.Image):
            pil_images.append(pad_image_pil_to_square(img.convert("RGB")))
        elif hasattr(img, 'name'):
            pil_images.append(pad_image_pil_to_square(Image.open(img.name).convert("RGB")))
    ref_images_pil_list = [pil_images]
    
    output = pipe(
        prompts=[prompt],
        negative_prompt=NEGATIVE_PROMPT,
        ref_images=ref_images_pil_list,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        image_guidance_scale=float(image_guidance_scale),
        seed=int(seed),
        timestep_shift=7.0,
        task="multiid",
    )
    
    return process_output(output, output_path)


def run_iv2v_edit(variant, ref_images, video_file, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed):
    """图像+视频编辑 (上下文 V2V)"""
    pipe = load_pipeline(variant)
    
    if ref_images is None or len(ref_images) == 0 or video_file is None:
        return None
    
    output_path = "outputs/iv2v_edit_output.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    # 处理参考图像 (Gallery 返回的是 [(filepath, caption), ...] 或 [filepath, ...])
    pil_images = []
    for img in ref_images:
        if isinstance(img, tuple):
            # Gallery format: (filepath, caption)
            pil_images.append(pad_image_pil_to_square(Image.open(img[0]).convert("RGB")))
        elif isinstance(img, str):
            pil_images.append(pad_image_pil_to_square(Image.open(img).convert("RGB")))
        elif isinstance(img, Image.Image):
            pil_images.append(pad_image_pil_to_square(img.convert("RGB")))
        elif hasattr(img, 'name'):
            pil_images.append(pad_image_pil_to_square(Image.open(img.name).convert("RGB")))
    ref_images_pil_list = [pil_images]
    
    output = pipe(
        prompts=[prompt],
        negative_prompt=NEGATIVE_PROMPT,
        ref_images=ref_images_pil_list,
        cond_video_path=video_file,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        image_guidance_scale=float(image_guidance_scale),
        seed=int(seed),
        timestep_shift=7.0,
        task="i+v2v_edit",
    )
    
    return process_output(output, output_path)


# ==================== 构建 Gradio 界面 ====================

def create_header():
    """创建页面头部"""
    return gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">🎬 UniVideo 演示</h1>
        <p style="color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.1em;">统一的视频生成与理解模型</p>
        <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 8px;">
            <p style="color: white; margin: 0; font-size: 1em;">
                📺 <strong>AI 技术分享频道</strong> - 
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="color: #ffeb3b; text-decoration: none;">
                    https://www.youtube.com/@rongyikanshijie-ai
                </a>
            </p>
        </div>
    </div>
    """)


def create_variant_selector():
    """创建模型变体选择器"""
    return gr.Radio(
        choices=[("变体1 (隐藏状态)", "variant1"), ("变体2 (查询向量)", "variant2")],
        value="variant1",
        label="模型变体",
        info="变体1: 图像/视频/文本 → MLLM → 最后一层隐藏状态 → MMDiT | 变体2: 图像/视频/文本/查询 → MLLM → 文本+查询隐藏状态 → MMDiT"
    )


def create_understanding_tab():
    """创建视频理解标签页"""
    with gr.TabItem("🎥 视频理解"):
        gr.Markdown("### 视频理解\n上传视频，让 AI 描述视频内容")
        
        with gr.Row():
            with gr.Column():
                variant = create_variant_selector()
                video_input = gr.Video(label="输入视频")
                prompt = gr.Textbox(
                    label="提示词",
                    value="Describe this video in detail",
                    lines=2
                )
                max_video_frames = gr.Slider(
                    label="最大视频帧数 (帧数越少显存占用越小)",
                    minimum=5,
                    maximum=65,
                    step=4,
                    value=17,
                    info="建议: 17帧约需40GB显存，33帧约需80GB显存"
                )
                seed = gr.Number(label="随机种子", value=42)
                run_btn = gr.Button("🚀 开始分析", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(label="分析结果", lines=10)
        
        # 示例
        gr.Examples(
            examples=[
                [os.path.join(DEMO_PATH, "understanding/1.mp4"), "Describe this video in detail", 17, 42]
            ],
            inputs=[video_input, prompt, max_video_frames, seed],
            label="示例"
        )
        
        run_btn.click(
            fn=run_understanding,
            inputs=[variant, video_input, prompt, max_video_frames, seed],
            outputs=output_text
        )


def create_t2v_tab():
    """创建文本生成视频标签页"""
    with gr.TabItem("📝 文本生成视频"):
        gr.Markdown("### 文本生成视频\n根据文字描述生成视频")
        
        with gr.Row():
            with gr.Column():
                variant = create_variant_selector()
                prompt = gr.Textbox(
                    label="提示词",
                    value="a stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
                    lines=4
                )
                
                with gr.Row():
                    height = gr.Number(label="高度", value=480)
                    width = gr.Number(label="宽度", value=854)
                    num_frames = gr.Number(label="帧数", value=61)
                
                with gr.Row():
                    num_steps = gr.Slider(label="推理步数", minimum=10, maximum=100, value=30, step=1)
                    guidance_scale = gr.Slider(label="引导系数", minimum=1, maximum=20, value=6.0, step=0.5)
                
                with gr.Row():
                    image_guidance_scale = gr.Slider(label="图像引导系数", minimum=0, maximum=10, value=1.0, step=0.5)
                    seed = gr.Number(label="随机种子", value=42)
                
                run_btn = gr.Button("🚀 开始生成", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="生成结果")
        
        # 示例
        gr.Examples(
            examples=[
                ["a stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.", 480, 854, 61, 30, 6.0, 1.0, 42]
            ],
            inputs=[prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            label="示例"
        )
        
        run_btn.click(
            fn=run_t2v,
            inputs=[variant, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            outputs=output_video
        )


def create_i2v_tab():
    """创建图像生成视频标签页"""
    with gr.TabItem("🖼️ 图像生成视频"):
        gr.Markdown("### 图像生成视频\n根据图像和文字描述生成视频")
        
        with gr.Row():
            with gr.Column():
                variant = create_variant_selector()
                image_input = gr.Image(label="输入图像", type="filepath")
                prompt = gr.Textbox(
                    label="提示词",
                    value="The video shows a small capybara wearing round glasses, holding a book titled 'UniVideo' on its cover. The capybara keeps the book lifted in front of its face, gently turning pages as it reads, its head making small, focused nods that match the rhythm of careful study. Its posture remains steady as both paws grip the book, and its ears tilt slightly with each subtle movement. Soft, warm lighting and a simple blurred background stay secondary to the close-up focus on the capybara, its glasses, and the reading motion.",
                    lines=4
                )
                
                with gr.Row():
                    height = gr.Number(label="高度", value=480)
                    width = gr.Number(label="宽度", value=854)
                    num_frames = gr.Number(label="帧数", value=129)
                
                with gr.Row():
                    num_steps = gr.Slider(label="推理步数", minimum=10, maximum=100, value=30, step=1)
                    guidance_scale = gr.Slider(label="引导系数", minimum=1, maximum=20, value=5.0, step=0.5)
                
                with gr.Row():
                    image_guidance_scale = gr.Slider(label="图像引导系数", minimum=0, maximum=10, value=1.0, step=0.5)
                    seed = gr.Number(label="随机种子", value=42)
                
                run_btn = gr.Button("🚀 开始生成", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="生成结果")
        
        # 示例
        gr.Examples(
            examples=[
                [os.path.join(DEMO_PATH, "i2v/1.png"), "The video shows a small capybara wearing round glasses, holding a book titled 'UniVideo' on its cover. The capybara keeps the book lifted in front of its face, gently turning pages as it reads, its head making small, focused nods that match the rhythm of careful study. Its posture remains steady as both paws grip the book, and its ears tilt slightly with each subtle movement. Soft, warm lighting and a simple blurred background stay secondary to the close-up focus on the capybara, its glasses, and the reading motion.", 480, 854, 129, 30, 5.0, 1.0, 42]
            ],
            inputs=[image_input, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            label="示例"
        )
        
        run_btn.click(
            fn=run_i2v,
            inputs=[variant, image_input, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            outputs=output_video
        )


def create_i2i_edit_tab():
    """创建图像编辑标签页"""
    with gr.TabItem("✏️ 图像编辑"):
        gr.Markdown("### 图像编辑\n根据文字指令编辑图像")
        
        with gr.Row():
            with gr.Column():
                variant = create_variant_selector()
                image_input = gr.Image(label="输入图像", type="filepath")
                prompt = gr.Textbox(
                    label="编辑提示词",
                    value="Change the background to dessert.",
                    lines=2
                )
                
                with gr.Row():
                    height = gr.Number(label="高度", value=480)
                    width = gr.Number(label="宽度", value=832)
                
                with gr.Row():
                    num_steps = gr.Slider(label="推理步数", minimum=10, maximum=100, value=50, step=1)
                    guidance_scale = gr.Slider(label="引导系数", minimum=1, maximum=20, value=7.0, step=0.5)
                
                with gr.Row():
                    image_guidance_scale = gr.Slider(label="图像引导系数", minimum=0, maximum=10, value=2.0, step=0.5)
                    seed = gr.Number(label="随机种子", value=42)
                
                run_btn = gr.Button("🚀 开始编辑", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="编辑结果")
        
        # 示例
        gr.Examples(
            examples=[
                [os.path.join(DEMO_PATH, "i2i_edit/1.jpg"), "Change the background to dessert.", 480, 832, 50, 7.0, 2.0, 42]
            ],
            inputs=[image_input, prompt, height, width, num_steps, guidance_scale, image_guidance_scale, seed],
            label="示例"
        )
        
        run_btn.click(
            fn=run_i2i_edit,
            inputs=[variant, image_input, prompt, height, width, num_steps, guidance_scale, image_guidance_scale, seed],
            outputs=output_image
        )


def create_v2v_edit_tab():
    """创建视频编辑标签页"""
    with gr.TabItem("🎬 视频编辑"):
        gr.Markdown("### 视频编辑\n根据文字指令编辑视频")
        
        with gr.Row():
            with gr.Column():
                variant = create_variant_selector()
                video_input = gr.Video(label="输入视频")
                prompt = gr.Textbox(
                    label="编辑提示词",
                    value="Change the man to look like he is sculpted from chocolate.",
                    lines=2
                )
                
                with gr.Row():
                    height = gr.Number(label="高度", value=480)
                    width = gr.Number(label="宽度", value=854)
                    num_frames = gr.Number(label="帧数", value=129)
                
                with gr.Row():
                    num_steps = gr.Slider(label="推理步数", minimum=10, maximum=100, value=50, step=1)
                    guidance_scale = gr.Slider(label="引导系数", minimum=1, maximum=20, value=7.0, step=0.5)
                
                with gr.Row():
                    image_guidance_scale = gr.Slider(label="图像引导系数", minimum=0, maximum=10, value=2.0, step=0.5)
                    seed = gr.Number(label="随机种子", value=42)
                
                run_btn = gr.Button("🚀 开始编辑", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="编辑结果")
        
        # 示例
        gr.Examples(
            examples=[
                [os.path.join(DEMO_PATH, "v2v_edit/video.mp4"), "Change the man to look like he is sculpted from chocolate.", 480, 854, 129, 50, 7.0, 2.0, 42]
            ],
            inputs=[video_input, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            label="示例"
        )
        
        run_btn.click(
            fn=run_v2v_edit,
            inputs=[variant, video_input, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            outputs=output_video
        )


def create_multiid_tab():
    """创建多身份上下文生成标签页"""
    with gr.TabItem("👥 多身份生成"):
        gr.Markdown("### 多身份上下文生成\n上传多张参考图像，生成包含这些身份的视频")
        
        with gr.Row():
            with gr.Column():
                variant = create_variant_selector()
                ref_images = gr.Gallery(
                    label="参考图像 (可上传多张，点击上传或拖拽)",
                    columns=3,
                    rows=1,
                    height="auto",
                    interactive=True,
                    type="filepath"
                )
                prompt = gr.Textbox(
                    label="提示词",
                    value="A man with short, light brown hair and light skin, now dressed in a vibrant Hawaiian shirt with a colorful floral pattern, sits comfortably on a beach lounge chair. On his right shoulder, a fluffy, yellow Pikachu with a small detective hat perches, looking alertly at the camera. The man holds an ice cream cone piled high with vanilla ice cream and colorful sprinkles, taking a bite with a relaxed, happy expression. His smile is gentle and content, reflecting the ease of the moment. The camera slowly circles around them, capturing the leisurely scene from various perspectives.",
                    lines=4
                )
                
                with gr.Row():
                    height = gr.Number(label="高度", value=480)
                    width = gr.Number(label="宽度", value=832)
                    num_frames = gr.Number(label="帧数", value=129)
                
                with gr.Row():
                    num_steps = gr.Slider(label="推理步数", minimum=10, maximum=100, value=50, step=1)
                    guidance_scale = gr.Slider(label="引导系数", minimum=1, maximum=20, value=5.0, step=0.5)
                
                with gr.Row():
                    image_guidance_scale = gr.Slider(label="图像引导系数", minimum=0, maximum=10, value=3.0, step=0.5)
                    seed = gr.Number(label="随机种子", value=42)
                
                run_btn = gr.Button("🚀 开始生成", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="生成结果")
        
        # 示例
        example_images = [
            os.path.join(DEMO_PATH, "in-context-generation/1.png"),
            os.path.join(DEMO_PATH, "in-context-generation/2.png"),
            os.path.join(DEMO_PATH, "in-context-generation/3.jpg")
        ]
        gr.Examples(
            examples=[
                [
                    example_images,
                    "A man with short, light brown hair and light skin, now dressed in a vibrant Hawaiian shirt with a colorful floral pattern, sits comfortably on a beach lounge chair. On his right shoulder, a fluffy, yellow Pikachu with a small detective hat perches, looking alertly at the camera. The man holds an ice cream cone piled high with vanilla ice cream and colorful sprinkles, taking a bite with a relaxed, happy expression. His smile is gentle and content, reflecting the ease of the moment. The camera slowly circles around them, capturing the leisurely scene from various perspectives.",
                    480, 832, 129, 50, 5.0, 3.0, 42
                ]
            ],
            inputs=[ref_images, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            label="示例"
        )
        
        run_btn.click(
            fn=run_multiid,
            inputs=[variant, ref_images, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            outputs=output_video
        )


def create_iv2v_edit_tab():
    """创建图像+视频编辑标签页"""
    with gr.TabItem("🔄 上下文视频编辑"):
        gr.Markdown("### 上下文视频编辑 (图像+视频)\n使用参考图像对视频进行编辑，例如身份替换")
        
        with gr.Row():
            with gr.Column():
                variant = create_variant_selector()
                ref_images = gr.Gallery(
                    label="参考图像 (可上传多张，点击上传或拖拽)",
                    columns=3,
                    rows=1,
                    height="auto",
                    interactive=True,
                    type="filepath"
                )
                video_input = gr.Video(label="输入视频")
                prompt = gr.Textbox(
                    label="编辑提示词",
                    value="Use the man's face in the reference image to replace the man's face in the video.",
                    lines=2
                )
                
                with gr.Row():
                    height = gr.Number(label="高度", value=480)
                    width = gr.Number(label="宽度", value=832)
                    num_frames = gr.Number(label="帧数 (导出24fps: 137帧≈5.7s)", value=137)
                
                with gr.Row():
                    num_steps = gr.Slider(label="推理步数", minimum=10, maximum=100, value=50, step=1)
                    guidance_scale = gr.Slider(label="引导系数", minimum=1, maximum=20, value=7.0, step=0.5)
                
                with gr.Row():
                    image_guidance_scale = gr.Slider(label="图像引导系数", minimum=0, maximum=10, value=2.0, step=0.5)
                    seed = gr.Number(label="随机种子", value=42)
                
                run_btn = gr.Button("🚀 开始编辑", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="编辑结果")
        
        # 示例
        gr.Examples(
            examples=[
                [
                    [os.path.join(DEMO_PATH, "in-context-v2v/id_swap/ID.jpeg")],
                    os.path.join(DEMO_PATH, "in-context-v2v/id_swap/origin.mp4"),
                    "Use the man's face in the reference image to replace the man's face in the video.",
                    480, 832, 137, 50, 7.0, 2.0, 42
                ]
            ],
            inputs=[ref_images, video_input, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            label="示例"
        )
        
        run_btn.click(
            fn=run_iv2v_edit,
            inputs=[variant, ref_images, video_input, prompt, height, width, num_frames, num_steps, guidance_scale, image_guidance_scale, seed],
            outputs=output_video
        )


def create_app():
    """创建 Gradio 应用"""
    with gr.Blocks(
        title="UniVideo 演示",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """
    ) as app:
        create_header()
        
        gr.Markdown("""
        ### 📖 使用说明
        
        1. **选择模型变体**: 在每个任务标签页中，您可以选择使用变体1或变体2
           - **变体1**: 图像/视频/文本 → MLLM → 最后一层隐藏状态 → MMDiT
           - **变体2**: 图像/视频/文本/查询 → MLLM → 文本+查询隐藏状态 → MMDiT
        
        2. **上传输入**: 根据任务类型上传图像或视频
        
        3. **设置参数**: 调整生成参数（可选）
        
        4. **点击生成**: 等待结果生成
        
        ---
        """)
        
        with gr.Tabs():
            create_understanding_tab()
            create_t2v_tab()
            create_i2v_tab()
            create_i2i_edit_tab()
            create_v2v_edit_tab()
            create_multiid_tab()
            create_iv2v_edit_tab()
        
        gr.Markdown("""
        ---
        ### ⚠️ 注意事项
        
        - 首次运行时，模型加载可能需要较长时间
        - 视频生成需要大量显存，建议使用具有 24GB+ 显存的 GPU
        - 切换模型变体会重新加载模型
        
        ---
        <div style="text-align: center; color: #888; padding: 20px;">
            Powered by UniVideo | 
            <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">AI 技术分享频道</a>
        </div>
        """)
    
    return app


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    # 启动应用
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
