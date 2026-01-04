#!/usr/bin/env python3
"""
TurboDiffusion Gradio Web Application
基于 Gradio 的视频生成 Web 应用

支持两种模式:
- 文生视频 (T2V): 根据文字提示生成视频
- 图生视频 (I2V): 根据图片和文字提示生成视频
"""

import os
import sys
import math
import tempfile
import uuid
from pathlib import Path

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置 PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "turbodiffusion"))

import torch
import gradio as gr
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm
import torchvision.transforms.v2 as T

# ============================================================
# 在导入 SLA 之前，先准备 PyTorch 替代实现
# ============================================================
def mean_pool_pytorch(x, BLK):
    """纯 PyTorch 实现的 mean_pool，替代 Triton 版本"""
    B, H, L, D = x.shape
    num_blocks = (L + BLK - 1) // BLK
    pad_size = num_blocks * BLK - L
    
    if pad_size > 0:
        x_padded = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
    else:
        x_padded = x
    
    x_reshaped = x_padded.view(B, H, num_blocks, BLK, D)
    # 对于部分填充的最后一个块，需要正确计算均值
    x_mean = x_reshaped.float().mean(dim=3)
    return x_mean.to(x.dtype)

# 导入 SLA 模块并替换函数
import SLA.utils as sla_utils
sla_utils.mean_pool = mean_pool_pytorch

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from inference.modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

# ============================================================
# 全局模型变量
# ============================================================
t2v_model = None
i2v_high_noise_model = None
i2v_low_noise_model = None
tokenizer = None
text_encoder_path = "checkpoints/models_t5_umt5-xxl-enc-bf16.pth"

# 当前活跃的模型模式 (用于管理 GPU 内存)
current_mode = None  # "t2v" 或 "i2v"

# ============================================================
# 模型加载函数
# ============================================================
def load_models():
    """启动时加载所有模型到 GPU"""
    global t2v_model, i2v_high_noise_model, i2v_low_noise_model, tokenizer
    
    log.info("正在加载模型...")
    
    # 创建参数对象
    class Args:
        model = "Wan2.1-1.3B"
        attention_type = "sagesla"  # 检查点需要 SageSLA 注意力
        sla_topk = 0.1
        quant_linear = False  # H20 有 96GB 显存，不需要量化
        default_norm = False
    
    args = Args()
    
    # 加载 T2V 模型 (直接放 GPU)
    log.info("加载 T2V 模型...")
    t2v_model = create_model(
        dit_path="checkpoints/TurboWan2.1-T2V-1.3B-480P.pth",
        args=args
    ).cuda().eval()
    torch.cuda.synchronize()
    log.success("T2V 模型加载完成")
    
    # 加载 I2V 模型 (直接放 GPU)
    log.info("加载 I2V 高噪声模型...")
    args.model = "Wan2.2-A14B"
    i2v_high_noise_model = create_model(
        dit_path="checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth",
        args=args
    ).cuda().eval()
    torch.cuda.synchronize()
    log.success("I2V 高噪声模型加载完成")
    
    log.info("加载 I2V 低噪声模型...")
    i2v_low_noise_model = create_model(
        dit_path="checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth",
        args=args
    ).cuda().eval()
    torch.cuda.synchronize()
    log.success("I2V 低噪声模型加载完成")
    
    # 加载 VAE
    log.info("加载 VAE...")
    tokenizer = Wan2pt1VAEInterface(vae_pth="checkpoints/Wan2.1_VAE.pth")
    log.success("VAE 加载完成")
    
    log.success("所有模型加载完成！")

# ============================================================
# T2V 生成函数
# ============================================================
def generate_t2v(
    prompt: str,
    num_frames: int,
    num_steps: int,
    resolution: str,
    aspect_ratio: str,
    sigma_max: float,
    sla_topk: float,
    seed: int,
    progress=gr.Progress(track_tqdm=True)
):
    """文生视频生成"""
    global t2v_model, tokenizer
    
    if not prompt.strip():
        raise gr.Error("请输入提示词")
    
    log.info(f"正在为提示词计算嵌入: {prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)
    clear_umt5_memory()
    
    w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
    
    condition = {"crossattn_emb": text_emb.to(**tensor_kwargs)}
    
    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]
    
    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(int(seed))
    
    init_noise = torch.randn(
        1,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )
    
    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1
    
    # 模型已在 GPU 上，直接推理
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="采样中", total=total_steps)):
        with torch.no_grad():
            v_pred = t2v_model(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )
    
    samples = x.float()
    
    with torch.no_grad():
        video = tokenizer.decode(samples)
    
    video_output = video.float().cpu()
    video_output = (1.0 + video_output.clamp(-1, 1)) / 2.0
    
    # 清理临时张量
    del x, init_noise, samples, text_emb, condition
    torch.cuda.empty_cache()
    
    # 保存视频
    output_path = os.path.join(tempfile.gettempdir(), f"t2v_{uuid.uuid4().hex[:8]}.mp4")
    save_image_or_video(rearrange(video_output, "b c t h w -> c t h (b w)"), output_path, fps=16)
    
    return output_path

# ============================================================
# I2V 生成函数
# ============================================================
def generate_i2v(
    image: Image.Image,
    prompt: str,
    num_frames: int,
    num_steps: int,
    resolution: str,
    aspect_ratio: str,
    adaptive_resolution: bool,
    use_ode: bool,
    sigma_max: float,
    boundary: float,
    sla_topk: float,
    seed: int,
    progress=gr.Progress(track_tqdm=True)
):
    """图生视频生成"""
    global i2v_high_noise_model, i2v_low_noise_model, tokenizer
    
    if image is None:
        raise gr.Error("请上传输入图片")
    if not prompt.strip():
        raise gr.Error("请输入提示词")
    
    log.info(f"正在为提示词计算嵌入: {prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)
    clear_umt5_memory()
    
    # 处理分辨率
    if adaptive_resolution:
        base_w, base_h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        max_resolution_area = base_w * base_h
        
        orig_w, orig_h = image.size
        image_aspect_ratio = orig_h / orig_w
        
        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)
        
        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride
    else:
        w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
    
    F = num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)
    
    # 预处理图像
    image_transforms = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = image_transforms(image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)
    
    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)], dim=2
        )
        encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode
        torch.cuda.empty_cache()
    
    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0
    
    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    
    condition = {
        "crossattn_emb": text_emb.to(**tensor_kwargs),
        "y_B_C_T_H_W": y
    }
    
    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
    
    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(int(seed))

    init_noise = torch.randn(
        1,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )
    
    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1
    
    # 模型已在 GPU 上
    # I2V 使用两个模型：先用高噪声模型，到达边界后切换到低噪声模型
    net = i2v_high_noise_model
    switched = False
    
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="采样中", total=total_steps)):
        if t_cur.item() < boundary and not switched:
            net = i2v_low_noise_model
            switched = True
            log.info("切换到低噪声模型")
        
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)
            
            if use_ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )
    
    samples = x.float()
    
    with torch.no_grad():
        video = tokenizer.decode(samples)
    
    video_output = video.float().cpu()
    video_output = (1.0 + video_output.clamp(-1, 1)) / 2.0
    
    # 清理临时张量
    del x, init_noise, samples, text_emb, condition, encoded_latents, y, msk
    torch.cuda.empty_cache()
    
    # 保存视频
    output_path = os.path.join(tempfile.gettempdir(), f"i2v_{uuid.uuid4().hex[:8]}.mp4")
    save_image_or_video(rearrange(video_output, "b c t h w -> c t h (b w)"), output_path, fps=16)
    
    return output_path

# ============================================================
# 加载示例数据
# ============================================================
def load_t2v_examples():
    """加载 T2V 示例"""
    prompts_file = Path("assets/t2v_inputs/prompts.txt")
    if prompts_file.exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        return [[p] for p in prompts[:10]]  # 最多10个示例
    return []

def load_i2v_examples():
    """加载 I2V 示例"""
    examples = []
    prompts_file = Path("assets/i2v_inputs/prompts.txt")
    images_dir = Path("assets/i2v_inputs")
    
    if prompts_file.exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        for i, prompt in enumerate(prompts):
            image_path = images_dir / f"i2v_input_{i}.jpg"
            if image_path.exists():
                examples.append([str(image_path), prompt])
    
    return examples

# ============================================================
# Gradio 界面
# ============================================================
def create_ui():
    """创建 Gradio 界面"""
    
    # YouTube 频道信息
    youtube_html = """
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%); border-radius: 10px; margin-bottom: 20px;">
        <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="text-decoration: none; color: white;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="white">
                    <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
                </svg>
                <span style="font-size: 18px; font-weight: bold;">AI 技术分享频道</span>
            </div>
            <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">点击订阅，获取更多 AI 技术教程</p>
        </a>
    </div>
    """
    
    with gr.Blocks(title="TurboDiffusion 视频生成", theme=gr.themes.Soft()) as demo:
        gr.HTML(youtube_html)
        gr.Markdown("# 🚀 TurboDiffusion 视频生成")
        gr.Markdown("基于 TurboDiffusion 的高速视频生成，支持文生视频和图生视频两种模式。")
        
        with gr.Tabs():
            # ========== T2V Tab ==========
            with gr.TabItem("📝 文生视频 (T2V)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2v_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入描述视频内容的提示词...",
                            lines=4
                        )
                        
                        with gr.Row():
                            t2v_resolution = gr.Dropdown(
                                choices=["480p", "720p"],
                                value="480p",
                                label="分辨率"
                            )
                            t2v_aspect_ratio = gr.Dropdown(
                                choices=["16:9", "9:16", "4:3", "3:4", "1:1"],
                                value="16:9",
                                label="宽高比"
                            )
                        
                        with gr.Row():
                            t2v_num_frames = gr.Slider(
                                minimum=17,
                                maximum=129,
                                value=81,
                                step=8,
                                label="帧数"
                            )
                            t2v_num_steps = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=4,
                                step=1,
                                label="采样步数"
                            )
                        
                        with gr.Accordion("高级选项", open=False):
                            t2v_sigma_max = gr.Slider(
                                minimum=10,
                                maximum=200,
                                value=80,
                                step=10,
                                label="Sigma Max (初始噪声强度)"
                            )
                            t2v_sla_topk = gr.Slider(
                                minimum=0.05,
                                maximum=0.3,
                                value=0.1,
                                step=0.05,
                                label="SLA Top-K 比例"
                            )
                            t2v_seed = gr.Number(
                                value=0,
                                label="随机种子",
                                precision=0
                            )
                        
                        t2v_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        t2v_output = gr.Video(label="生成结果")
                
                # T2V 示例
                t2v_examples = load_t2v_examples()
                if t2v_examples:
                    gr.Examples(
                        examples=t2v_examples,
                        inputs=[t2v_prompt],
                        label="示例提示词"
                    )
                
                t2v_btn.click(
                    fn=generate_t2v,
                    inputs=[
                        t2v_prompt,
                        t2v_num_frames,
                        t2v_num_steps,
                        t2v_resolution,
                        t2v_aspect_ratio,
                        t2v_sigma_max,
                        t2v_sla_topk,
                        t2v_seed
                    ],
                    outputs=t2v_output
                )
            
            # ========== I2V Tab ==========
            with gr.TabItem("🖼️ 图生视频 (I2V)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_image = gr.Image(
                            label="输入图片",
                            type="pil"
                        )
                        i2v_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入描述视频动作的提示词...",
                            lines=4
                        )
                        
                        with gr.Row():
                            i2v_resolution = gr.Dropdown(
                                choices=["480p", "720p"],
                                value="720p",
                                label="分辨率"
                            )
                            i2v_aspect_ratio = gr.Dropdown(
                                choices=["16:9", "9:16", "4:3", "3:4", "1:1"],
                                value="16:9",
                                label="宽高比"
                            )
                        
                        with gr.Row():
                            i2v_num_frames = gr.Slider(
                                minimum=17,
                                maximum=129,
                                value=81,
                                step=8,
                                label="帧数"
                            )
                            i2v_num_steps = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=4,
                                step=1,
                                label="采样步数"
                            )
                        
                        with gr.Row():
                            i2v_adaptive_resolution = gr.Checkbox(
                                value=True,
                                label="自适应分辨率"
                            )
                            i2v_use_ode = gr.Checkbox(
                                value=True,
                                label="使用 ODE 采样 (更锐利)"
                            )
                        
                        with gr.Accordion("高级选项", open=False):
                            i2v_sigma_max = gr.Slider(
                                minimum=50,
                                maximum=400,
                                value=200,
                                step=25,
                                label="Sigma Max (初始噪声强度)"
                            )
                            i2v_boundary = gr.Slider(
                                minimum=0.5,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="模型切换边界"
                            )
                            i2v_sla_topk = gr.Slider(
                                minimum=0.05,
                                maximum=0.3,
                                value=0.1,
                                step=0.05,
                                label="SLA Top-K 比例"
                            )
                            i2v_seed = gr.Number(
                                value=0,
                                label="随机种子",
                                precision=0
                            )
                        
                        i2v_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        i2v_output = gr.Video(label="生成结果")
                
                # I2V 示例
                i2v_examples = load_i2v_examples()
                if i2v_examples:
                    gr.Examples(
                        examples=i2v_examples,
                        inputs=[i2v_image, i2v_prompt],
                        label="示例图片和提示词"
                    )
                
                i2v_btn.click(
                    fn=generate_i2v,
                    inputs=[
                        i2v_image,
                        i2v_prompt,
                        i2v_num_frames,
                        i2v_num_steps,
                        i2v_resolution,
                        i2v_aspect_ratio,
                        i2v_adaptive_resolution,
                        i2v_use_ode,
                        i2v_sigma_max,
                        i2v_boundary,
                        i2v_sla_topk,
                        i2v_seed
                    ],
                    outputs=i2v_output
                )
        
        gr.Markdown("""
        ---
        ### 使用说明
        - **文生视频**: 输入文字提示词，模型将根据描述生成对应的视频
        - **图生视频**: 上传一张图片并输入提示词，模型将基于图片内容生成动态视频
        - **采样步数**: 步数越多质量越好，但生成时间也越长 (推荐 4 步)
        - **SLA Top-K**: 值越大生成质量越好，但速度越慢 (推荐 0.1-0.15)
        """)
    
    return demo

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    # 加载模型
    load_models()
    
    # 创建并启动界面
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
