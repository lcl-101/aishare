#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Image Fun - Gradio Web UI
基于 examples/qwenimage_fun/predict_t2i_control.py 和 examples/qwenimage_fun/predict_i2i_inpaint.py
"""

import os
import sys
import gc
import torch
import gradio as gr
from PIL import Image
import numpy as np

from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler

# 添加项目根目录到 sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKLQwenImage,
                               Qwen2_5_VLForConditionalGeneration,
                               Qwen2Tokenizer, QwenImageControlTransformer2DModel)
from videox_fun.pipeline import QwenImageControlPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import get_image_latent

# ====================== 全局配置 ======================
# GPU 内存模式
GPU_MEMORY_MODE = "model_cpu_offload_and_qfloat8"

# 配置文件路径
CONFIG_PATH = "config/qwenimage/qwenimage_control.yaml"

# 模型路径 (使用 checkpoints 文件夹)
MODEL_NAME = "checkpoints/Qwen-Image-2512"
TRANSFORMER_PATH = "checkpoints/Qwen-Image-2512-Fun-Controlnet-Union/Qwen-Image-2512-Fun-Controlnet-Union.safetensors"

# 数据类型
WEIGHT_DTYPE = torch.bfloat16

# 预设分辨率选项
RESOLUTION_OPTIONS = [
    "1728x992 (16:9 横版)",
    "992x1728 (9:16 竖版)",
    "1536x1024 (3:2 横版)",
    "1024x1536 (2:3 竖版)",
    "1024x1024 (1:1 正方形)",
    "1280x960 (4:3 横版)",
    "960x1280 (3:4 竖版)",
    "2048x1024 (2:1 超宽)",
    "1024x2048 (1:2 超高)",
]

# 采样器选项
SAMPLER_OPTIONS = ["Flow", "Flow_Unipc", "Flow_DPM++"]

# YouTube 频道信息
YOUTUBE_CHANNEL_NAME = "AI 技术分享频道"
YOUTUBE_CHANNEL_URL = "https://www.youtube.com/@rongyikanshijie-ai"

# ====================== 全局变量 ======================
pipeline = None
device = None

# ====================== 示例提示词 ======================
EXAMPLE_PROMPT = "画面中央是一位年轻女孩，她拥有一头令人印象深刻的亮紫色长发，发丝在海风中轻盈飘扬，营造出动感而唯美的效果。她的长发两侧各扎着黑色蝴蝶结发饰，增添了几分可爱与俏皮感。女孩身穿一袭纯白色无袖连衣裙，裙摆轻盈飘逸，与她清新的气质完美契合。她的妆容精致自然，淡粉色的唇妆和温柔的眼神流露出恬静优雅的气质。她单手叉腰，姿态自信从容，目光直视镜头，展现出既甜美又不失个性的魅力。背景是一片开阔的海景，湛蓝的海水在阳光照射下波光粼粼，闪烁着钻石般的光芒。天空呈现出清澈的蔚蓝色，点缀着几朵洁白的云朵，营造出晴朗明媚的夏日氛围。画面前景右下角可见粉紫色的小花丛和绿色植物，为整体构图增添了自然生机和色彩层次。整张照片色调明亮清新，紫色头发与白色裙装、蓝色海天形成鲜明而和谐的色彩对比。"

EXAMPLE_PROMPT_4 = "一只威风凛凛的花豹正面特写，锐利的眼睛直视前方，脸上布满独特的斑点花纹。背景是模糊的丛林绿叶，阳光透过树叶洒下斑驳的光影。"
EXAMPLE_NEGATIVE_PROMPT_4 = "模糊, 变形, 低质量"


def load_pipeline():
    """加载模型和管道"""
    global pipeline, device
    
    print("=" * 50)
    print("正在加载模型...")
    print("=" * 50)
    
    # 设置设备
    device = set_multi_gpus_devices(1, 1)
    
    # 加载配置
    config = OmegaConf.load(CONFIG_PATH)
    
    # 加载 Transformer
    print(f"正在加载 Transformer: {MODEL_NAME}")
    transformer = QwenImageControlTransformer2DModel.from_pretrained(
        MODEL_NAME, 
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(WEIGHT_DTYPE)
    
    # 加载 ControlNet 权重
    if TRANSFORMER_PATH is not None and os.path.exists(TRANSFORMER_PATH):
        print(f"正在加载 ControlNet 权重: {TRANSFORMER_PATH}")
        if TRANSFORMER_PATH.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(TRANSFORMER_PATH)
        else:
            state_dict = torch.load(TRANSFORMER_PATH, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"缺失键: {len(m)}, 意外键: {len(u)}")
    
    # 加载 VAE
    print(f"正在加载 VAE: {MODEL_NAME}")
    vae = AutoencoderKLQwenImage.from_pretrained(
        MODEL_NAME, 
        subfolder="vae"
    ).to(WEIGHT_DTYPE)
    
    # 加载 Tokenizer 和 Text Encoder
    print(f"正在加载 Tokenizer 和 Text Encoder: {MODEL_NAME}")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer"
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE
    )
    
    # 加载 Scheduler (默认使用 Flow)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_NAME, 
        subfolder="scheduler"
    )
    
    # 创建 Pipeline
    pipeline = QwenImageControlPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
    
    # 应用 GPU 内存优化
    if GPU_MEMORY_MODE == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
        convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
        convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)
    
    print("=" * 50)
    print("模型加载完成！")
    print("=" * 50)
    
    return pipeline


def parse_resolution(resolution_str):
    """解析分辨率字符串"""
    # 格式: "1728x992 (16:9 横版)"
    res_part = resolution_str.split(" ")[0]
    width, height = map(int, res_part.split("x"))
    return height, width  # 返回 (height, width) 格式


def get_scheduler(sampler_name):
    """获取调度器"""
    scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    Chosen_Scheduler = scheduler_dict.get(sampler_name, FlowMatchEulerDiscreteScheduler)
    return Chosen_Scheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")


def generate_t2i_control(
    prompt,
    negative_prompt,
    control_image,
    resolution,
    sampler_name,
    guidance_scale,
    num_inference_steps,
    control_context_scale,
    seed,
    progress=gr.Progress(track_tqdm=True)
):
    """文生图 + Control 模式"""
    global pipeline, device
    
    if pipeline is None:
        return None, "错误：模型未加载，请稍候..."
    
    if control_image is None:
        return None, "错误：请上传控制图像"
    
    if not prompt.strip():
        return None, "错误：请输入提示词"
    
    try:
        # 解析分辨率
        height, width = parse_resolution(resolution)
        sample_size = [height, width]
        
        # 更新调度器
        pipeline.scheduler = get_scheduler(sampler_name)
        
        # 设置随机种子
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
        # 准备输入图像
        with torch.no_grad():
            # 无 inpaint 图像
            inpaint_image_input = torch.zeros([1, 3, sample_size[0], sample_size[1]])
            # 全白 mask
            mask_image_input = torch.ones([1, 1, sample_size[0], sample_size[1]]) * 255
            # 控制图像
            control_image_input = get_image_latent(control_image, sample_size=sample_size)[:, :, 0]
            
            # 生成图像
            sample = pipeline(
                prompt, 
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                true_cfg_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                image=inpaint_image_input,
                mask_image=mask_image_input,
                control_image=control_image_input,
                control_context_scale=control_context_scale
            ).images
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()
        gc.collect()
        
        result_image = sample[0]
        return result_image, f"生成成功！使用的种子: {seed}"
    
    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return None, f"生成失败: {str(e)}"


def generate_i2i_inpaint(
    prompt,
    negative_prompt,
    inpaint_image,
    mask_image,
    control_image,
    resolution,
    sampler_name,
    guidance_scale,
    num_inference_steps,
    control_context_scale,
    seed,
    progress=gr.Progress(track_tqdm=True)
):
    """图生图 + Inpaint 模式"""
    global pipeline, device
    
    if pipeline is None:
        return None, "错误：模型未加载，请稍候..."
    
    if inpaint_image is None:
        return None, "错误：请上传待修复图像"
    
    if mask_image is None:
        return None, "错误：请上传遮罩图像"
    
    if not prompt.strip():
        return None, "错误：请输入提示词"
    
    try:
        # 解析分辨率
        height, width = parse_resolution(resolution)
        sample_size = [height, width]
        
        # 更新调度器
        pipeline.scheduler = get_scheduler(sampler_name)
        
        # 设置随机种子
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
        # 准备输入图像
        with torch.no_grad():
            # Inpaint 图像
            inpaint_image_input = get_image_latent(inpaint_image, sample_size=sample_size)[:, :, 0]
            # Mask 图像
            mask_image_input = get_image_latent(mask_image, sample_size=sample_size)[:, :1, 0]
            # 控制图像 (可选)
            if control_image is not None:
                control_image_input = get_image_latent(control_image, sample_size=sample_size)[:, :, 0]
            else:
                control_image_input = torch.zeros([1, 3, sample_size[0], sample_size[1]])
            
            # 生成图像
            sample = pipeline(
                prompt, 
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                true_cfg_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                image=inpaint_image_input,
                mask_image=mask_image_input,
                control_image=control_image_input,
                control_context_scale=control_context_scale
            ).images
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()
        gc.collect()
        
        result_image = sample[0]
        return result_image, f"生成成功！使用的种子: {seed}"
    
    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return None, f"生成失败: {str(e)}"


def create_ui():
    """创建 Gradio UI"""
    
    # 准备示例图像
    example_control_images = []
    example_inpaint_images = []
    
    # 从 asset 文件夹获取示例
    asset_dir = "asset"
    if os.path.exists(asset_dir):
        # 控制图像示例
        if os.path.exists(os.path.join(asset_dir, "pose.jpg")):
            example_control_images.append(os.path.join(asset_dir, "pose.jpg"))
        
        # Inpaint 示例
        if os.path.exists(os.path.join(asset_dir, "8.png")):
            example_inpaint_images.append(os.path.join(asset_dir, "8.png"))
    
    # 从 checkpoints 文件夹获取更多示例
    controlnet_asset_dir = "checkpoints/Qwen-Image-2512-Fun-Controlnet-Union/asset"
    if os.path.exists(controlnet_asset_dir):
        for img_name in ["pose.jpg", "pose2.jpg", "canny.jpg", "depth.jpg", "hed.jpg", "scribble.jpg"]:
            img_path = os.path.join(controlnet_asset_dir, img_name)
            if os.path.exists(img_path):
                example_control_images.append(img_path)
        
        # Inpaint 示例
        for img_name in ["inpaint.jpg"]:
            img_path = os.path.join(controlnet_asset_dir, img_name)
            if os.path.exists(img_path):
                example_inpaint_images.append(img_path)
        
        # Mask 示例
        mask_path = os.path.join(controlnet_asset_dir, "mask.jpg")
    
    # 从 asset 文件夹获取 mask
    mask_example = os.path.join(asset_dir, "mask.png") if os.path.exists(os.path.join(asset_dir, "mask.png")) else None
    
    # CSS 样式
    css = """
    .youtube-banner {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 16px;
    }
    .youtube-banner a {
        color: white;
        text-decoration: underline;
        font-weight: bold;
    }
    .title-text {
        text-align: center;
        margin-bottom: 10px;
    }
    """
    
    with gr.Blocks(css=css, title="Qwen Image Fun - AI 图像生成") as demo:
        # YouTube 频道信息
        gr.HTML(f"""
        <div class="youtube-banner">
            📺 欢迎访问我的 YouTube 频道: <a href="{YOUTUBE_CHANNEL_URL}" target="_blank">{YOUTUBE_CHANNEL_NAME}</a> 
            | 更多 AI 教程和技术分享，欢迎订阅！
        </div>
        """)
        
        # 标题
        gr.Markdown("""
        # 🎨 Qwen Image Fun - AI 图像生成
        基于 Qwen-Image-2512 模型，支持文生图控制 (T2I Control) 和图像修复 (I2I Inpaint) 两种模式。
        """)
        
        with gr.Tabs():
            # ==================== 文生图 + Control 标签页 ====================
            with gr.TabItem("📝 文生图 + Control"):
                gr.Markdown("### 使用控制图像（姿势、边缘、深度等）生成图像")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        t2i_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入描述图像内容的提示词...",
                            lines=5,
                            value=""
                        )
                        t2i_negative_prompt = gr.Textbox(
                            label="负面提示词",
                            placeholder="输入不希望出现的内容...",
                            lines=2,
                            value=""
                        )
                        t2i_control_image = gr.Image(
                            label="控制图像（姿势图、边缘图、深度图等）",
                            type="filepath",
                            height=300
                        )
                        
                        with gr.Row():
                            t2i_resolution = gr.Dropdown(
                                label="输出分辨率",
                                choices=RESOLUTION_OPTIONS,
                                value=RESOLUTION_OPTIONS[0]
                            )
                            t2i_sampler = gr.Dropdown(
                                label="采样器",
                                choices=SAMPLER_OPTIONS,
                                value=SAMPLER_OPTIONS[0]
                            )
                        
                        with gr.Row():
                            t2i_guidance_scale = gr.Slider(
                                label="引导系数 (CFG Scale)",
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5
                            )
                            t2i_steps = gr.Slider(
                                label="推理步数",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5
                            )
                        
                        with gr.Row():
                            t2i_control_scale = gr.Slider(
                                label="控制强度",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.80,
                                step=0.05
                            )
                            t2i_seed = gr.Number(
                                label="随机种子 (-1 为随机)",
                                value=-1,
                                precision=0
                            )
                        
                        t2i_generate_btn = gr.Button("🎨 生成图像", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        t2i_output_image = gr.Image(
                            label="生成结果",
                            type="pil",
                            height=500
                        )
                        t2i_status = gr.Textbox(
                            label="状态信息",
                            interactive=False
                        )
                
                # 示例
                if example_control_images:
                    gr.Markdown("### 📌 示例")
                    t2i_examples = gr.Examples(
                        examples=[
                            [
                                (EXAMPLE_PROMPT_4 if i == 3 else EXAMPLE_PROMPT),
                                (EXAMPLE_NEGATIVE_PROMPT_4 if i == 3 else " "),
                                img,
                                RESOLUTION_OPTIONS[0],
                                "Flow",
                                4.0,
                                50,
                                0.80,
                                43,
                            ]
                            for i, img in enumerate(example_control_images[:4])
                        ],
                        inputs=[t2i_prompt, t2i_negative_prompt, t2i_control_image, t2i_resolution, 
                               t2i_sampler, t2i_guidance_scale, t2i_steps, t2i_control_scale, t2i_seed],
                        label="点击加载示例"
                    )
                
                # 绑定生成事件
                t2i_generate_btn.click(
                    fn=generate_t2i_control,
                    inputs=[t2i_prompt, t2i_negative_prompt, t2i_control_image, t2i_resolution,
                           t2i_sampler, t2i_guidance_scale, t2i_steps, t2i_control_scale, t2i_seed],
                    outputs=[t2i_output_image, t2i_status]
                )
            
            # ==================== 图生图 + Inpaint 标签页 ====================
            with gr.TabItem("🖌️ 图生图 + Inpaint"):
                gr.Markdown("### 使用原图和遮罩进行图像修复或重绘")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        i2i_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入描述图像内容的提示词...",
                            lines=5,
                            value=""
                        )
                        i2i_negative_prompt = gr.Textbox(
                            label="负面提示词",
                            placeholder="输入不希望出现的内容...",
                            lines=2,
                            value=""
                        )
                        
                        with gr.Row():
                            i2i_inpaint_image = gr.Image(
                                label="待修复图像",
                                type="filepath",
                                height=200
                            )
                            i2i_mask_image = gr.Image(
                                label="遮罩图像（白色区域将被重绘）",
                                type="filepath",
                                height=200
                            )
                        
                        i2i_control_image = gr.Image(
                            label="控制图像（可选，用于引导生成）",
                            type="filepath",
                            height=200
                        )
                        
                        with gr.Row():
                            i2i_resolution = gr.Dropdown(
                                label="输出分辨率",
                                choices=RESOLUTION_OPTIONS,
                                value=RESOLUTION_OPTIONS[0]
                            )
                            i2i_sampler = gr.Dropdown(
                                label="采样器",
                                choices=SAMPLER_OPTIONS,
                                value=SAMPLER_OPTIONS[0]
                            )
                        
                        with gr.Row():
                            i2i_guidance_scale = gr.Slider(
                                label="引导系数 (CFG Scale)",
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5
                            )
                            i2i_steps = gr.Slider(
                                label="推理步数",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5
                            )
                        
                        with gr.Row():
                            i2i_control_scale = gr.Slider(
                                label="控制强度",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.80,
                                step=0.05
                            )
                            i2i_seed = gr.Number(
                                label="随机种子 (-1 为随机)",
                                value=-1,
                                precision=0
                            )
                        
                        i2i_generate_btn = gr.Button("🖌️ 开始修复", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        i2i_output_image = gr.Image(
                            label="生成结果",
                            type="pil",
                            height=500
                        )
                        i2i_status = gr.Textbox(
                            label="状态信息",
                            interactive=False
                        )
                
                # 示例
                inpaint_example_path = "asset/8.png"
                mask_example_path = "asset/mask.png"
                pose_example_path = "asset/pose.jpg"
                
                if os.path.exists(inpaint_example_path) and os.path.exists(mask_example_path):
                    gr.Markdown("### 📌 示例")
                    i2i_examples = gr.Examples(
                        examples=[
                            [EXAMPLE_PROMPT, " ", inpaint_example_path, mask_example_path, 
                             pose_example_path if os.path.exists(pose_example_path) else None,
                             RESOLUTION_OPTIONS[0], "Flow", 4.0, 50, 0.80, 43]
                        ],
                        inputs=[i2i_prompt, i2i_negative_prompt, i2i_inpaint_image, i2i_mask_image,
                               i2i_control_image, i2i_resolution, i2i_sampler, i2i_guidance_scale,
                               i2i_steps, i2i_control_scale, i2i_seed],
                        label="点击加载示例"
                    )
                
                # 绑定生成事件
                i2i_generate_btn.click(
                    fn=generate_i2i_inpaint,
                    inputs=[i2i_prompt, i2i_negative_prompt, i2i_inpaint_image, i2i_mask_image,
                           i2i_control_image, i2i_resolution, i2i_sampler, i2i_guidance_scale,
                           i2i_steps, i2i_control_scale, i2i_seed],
                    outputs=[i2i_output_image, i2i_status]
                )
        
        # 使用说明
        gr.Markdown("""
        ---
        ### 📖 使用说明
        
        **文生图 + Control 模式:**
        1. 上传一张控制图像（如姿势图、边缘图、深度图等）
        2. 输入描述目标图像的提示词
        3. 调整参数后点击"生成图像"
        
        **图生图 + Inpaint 模式:**
        1. 上传待修复的原始图像
        2. 上传遮罩图像（白色区域表示需要重绘的部分）
        3. 可选上传控制图像来引导生成
        4. 输入提示词后点击"开始修复"
        
        **参数说明:**
        - **引导系数 (CFG Scale)**: 值越大，生成结果越接近提示词描述，但可能降低图像质量
        - **推理步数**: 步数越多，图像质量越高，但生成时间越长
        - **控制强度**: 控制图像对生成结果的影响程度
        - **随机种子**: 相同种子会生成相同结果，-1 为随机
        """)
    
    return demo


if __name__ == "__main__":
    # 启动时加载模型
    print("=" * 60)
    print("Qwen Image Fun - Gradio Web UI")
    print("=" * 60)
    
    load_pipeline()
    
    # 创建并启动 UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
