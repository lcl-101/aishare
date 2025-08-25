#!/usr/bin/env python3
"""
Vivid-VR Gradio WebUI
基于正常工作的 inference.py 创建的Web界面
"""

import os
import gc
import sys
import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
import tempfile
import shutil
from tqdm import tqdm
from PIL import Image
from transformers import T5EncoderModel
from diffusers import CogVideoXDPMScheduler, AutoencoderKLCogVideoX, CogVideoXVividVRTransformer3DModel, CogVideoXVividVRControlNetModel, CogVideoXVividVRControlNetPipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './VRDiT')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from VRDiT.cogvlm2 import CogVLM2_Captioner
from VRDiT.colorfix import adaptive_instance_normalization
from VRDiT.utils import VALID_IMAGE_EXTENSIONS, VALID_VIDEO_EXTENSIONS, free_memory, load_video, export_to_video, prepare_validation_prompts, prepare_tiling_infos_generator


class VividVRModel:
    def __init__(self):
        self.pipe = None
        self.captioner_model = None
        self.text_fixer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        
    def initialize_models(self, progress=gr.Progress()):
        """初始化所有模型组件"""
        if self.initialized:
            return "✅ 模型已经加载完成"
            
        try:
            progress(0.1, desc="开始加载模型...")
            
            # 配置路径
            ckpt_dir = './ckpts'
            cogvideox_ckpt_path = './ckpts/CogVideoX1.5-5B'
            cogvlm2_ckpt_path = './ckpts/cogvlm2-llama3-caption'
            vividvr_ckpt_path = os.path.join(ckpt_dir, 'Vivid-VR', 'ckpts', 'Vivid-VR')
            
            progress(0.15, desc="加载 CogVLM2 字幕生成器...")
            self.captioner_model = CogVLM2_Captioner(model_path=cogvlm2_ckpt_path)
            
            progress(0.25, desc="加载 T5 文本编码器...")
            text_encoder = T5EncoderModel.from_pretrained(
                cogvideox_ckpt_path,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16
            )
            text_encoder.requires_grad_(False)
            text_encoder.to(dtype=torch.bfloat16)
            
            progress(0.35, desc="加载 Transformer...")
            transformer = CogVideoXVividVRTransformer3DModel.from_pretrained(
                cogvideox_ckpt_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16
            )
            transformer.requires_grad_(False)
            transformer.to(dtype=torch.bfloat16)
            
            progress(0.45, desc="加载 VAE...")
            vae = AutoencoderKLCogVideoX.from_pretrained(
                cogvideox_ckpt_path,
                subfolder="vae"
            )
            vae.requires_grad_(False)
            vae.to(dtype=torch.bfloat16)
            vae.enable_slicing()
            vae.enable_tiling()
            
            progress(0.55, desc="加载 ControlNet...")
            controlnet = CogVideoXVividVRControlNetModel.from_transformer(
                transformer=transformer,
                num_layers=6,
            )
            controlnet.requires_grad_(False)
            controlnet.to(dtype=torch.bfloat16)
            
            progress(0.65, desc="加载调度器...")
            scheduler = CogVideoXDPMScheduler.from_pretrained(
                cogvideox_ckpt_path,
                subfolder="scheduler"
            )
            
            progress(0.75, desc="加载 Vivid-VR 权重...")
            transformer.connectors.load_state_dict(torch.load(os.path.join(vividvr_ckpt_path, "connectors.pt"), map_location='cpu'))
            transformer.control_feat_proj.load_state_dict(torch.load(os.path.join(vividvr_ckpt_path, "control_feat_proj.pt"), map_location='cpu'))
            transformer.control_patch_embed.load_state_dict(torch.load(os.path.join(vividvr_ckpt_path, "control_patch_embed.pt"), map_location='cpu'))
            
            load_model = CogVideoXVividVRControlNetModel.from_pretrained(vividvr_ckpt_path, subfolder="controlnet")
            controlnet.register_to_config(**load_model.config)
            controlnet.load_state_dict(load_model.state_dict())
            del load_model
            
            free_memory()
            
            progress(0.85, desc="创建推理管道...")
            self.pipe = CogVideoXVividVRControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=cogvideox_ckpt_path,
                scheduler=scheduler,
                vae=vae,
                transformer=transformer,
                controlnet=controlnet,
                text_encoder=text_encoder,
                torch_dtype=torch.bfloat16,
            )
            
            progress(0.95, desc="完成初始化...")
            self.initialized = True
            free_memory()
            
            progress(1.0, desc="模型加载完成！")
            return "✅ 所有模型加载完成，可以开始视频增强！"
            
        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}"

    def process_video(
        self, 
        input_video, 
        num_inference_steps=50, 
        guidance_scale=6, 
        tile_size=128, 
        upscale=0.0,
        use_dynamic_cfg=False,
        seed=42,
        progress=gr.Progress()
    ):
        """处理视频的主要函数"""
        if not self.initialized:
            return None, "❌ 请先点击'初始化模型'按钮加载模型"
            
        if input_video is None:
            return None, "❌ 请上传视频文件"
            
        try:
            progress(0.05, desc="开始处理视频...")
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, "input_video.mp4")
            output_path = os.path.join(temp_dir, "output_video.mp4")
            
            # 复制输入视频到临时目录
            shutil.copy2(input_video, input_path)
            
            progress(0.1, desc="加载和预处理视频...")
            # 获取视频信息
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 24
            cap.release()
            
            # 加载视频
            control_video = load_video(input_path)
            
            # 应用缩放
            if upscale == 0.:
                scale_factor = 1024. / min(control_video.size()[2], control_video.size()[3])
                control_video = F.interpolate(control_video, scale_factor=scale_factor, mode='bicubic').clip(0, 1)
                height, width = control_video.size()[2], control_video.size()[3]
            elif upscale != 1.0:
                control_video = F.interpolate(control_video, scale_factor=upscale, mode='bicubic').clip(0, 1)
                height, width = control_video.size()[2], control_video.size()[3]
            else:
                height, width = control_video.size()[2], control_video.size()[3]
                
            progress(0.15, desc=f"视频尺寸: {height}x{width}")
            
            # 帧数填充
            num_padding_frames = 0
            if (control_video.size(0) - 1) % 8 != 0:
                num_padding_frames = 8 - (control_video.size(0) - 1) % 8
                control_video = torch.cat([control_video, control_video[-1:].repeat(num_padding_frames, 1, 1, 1)], dim=0)
            
            # 准备推理参数 - 在生成字幕之前定义 gen_height 和 gen_width
            vae_scale_factor_spatial = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
            gen_height = 8 * math.ceil(height / 8) if height < tile_size * vae_scale_factor_spatial else height
            gen_width = 8 * math.ceil(width / 8) if width < tile_size * vae_scale_factor_spatial else width
            
            progress(0.25, desc="生成视频描述...")
            # 生成字幕 - 按照 inference.py 的方式
            video_for_caption = F.interpolate(control_video, size=(gen_height, gen_width), mode='bicubic')
            prompt_list, negative_prompt_list = prepare_validation_prompts(
                video_for_caption=video_for_caption,
                video_fps=fps,
                captioner_model=self.captioner_model,
                tile_size=tile_size * vae_scale_factor_spatial,
                tile_stride=(tile_size // 2) * vae_scale_factor_spatial,
                device=self.device
            )
            
            progress(0.35, desc=f"生成的描述: {prompt_list[0][:50] if prompt_list else 'N/A'}...")
            
            progress(0.4, desc="开始视频增强推理...")
            # 运行推理
            self.pipe.enable_model_cpu_offload(device=self.device)
            
            result = self.pipe(
                control_video=control_video,
                prompt=prompt_list,
                negative_prompt=negative_prompt_list,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                height=gen_height,
                width=gen_width,
                num_inference_steps=num_inference_steps,
                enable_spatial_tiling=True,
                tile_size=tile_size,
                tile_stride=tile_size // 2,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                output_type="np"
            )
            
            video = result.frames[0]
            progress(0.85, desc="后处理视频...")
            
            # 移除填充帧
            if num_padding_frames > 0:
                control_video = control_video[:-num_padding_frames]  # 移除填充帧
                video = video[:-num_padding_frames]
                
            # 移除前几帧（如果需要）
            if video.shape[0] % 4 == 0:
                video = video[3:]
                
            progress(0.9, desc="保存输出视频...")
            # 保存视频 - 直接传递 numpy 数组，函数会自动处理格式转换
            export_to_video(video, output_path, fps=fps)
            
            progress(1.0, desc="处理完成！")
            
            # 清理内存
            free_memory()
            
            return output_path, f"✅ 视频增强完成！\\n原始尺寸: {control_video.shape}\\n输出尺寸: {video.shape}\\n生成描述: {prompt_list[0] if prompt_list else 'N/A'}"
            
        except Exception as e:
            return None, f"❌ 处理失败: {str(e)}"

# 全局模型实例
model = VividVRModel()

def update_video_preview(video_file):
    """更新视频预览"""
    if video_file is None:
        return None
    return video_file

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="Vivid-VR 视频增强", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 Vivid-VR 视频增强系统
        
        上传视频文件，使用Vivid-VR模型进行智能增强，提升视频质量和细节。
        
        **使用步骤：**
        1. 首先点击"初始化模型"按钮加载所有模型组件
        2. 上传要增强的视频文件
        3. 调整推理参数（可选）
        4. 点击"开始增强"按钮处理视频
        """)
        
        # 模型初始化区域
        with gr.Row():
            with gr.Column():
                init_btn = gr.Button("🚀 初始化模型", variant="primary", scale=2)
                init_status = gr.Textbox(
                    label="初始化状态", 
                    value="⏳ 点击上方按钮开始加载模型...",
                    interactive=False
                )
        
        # 主要处理区域
        with gr.Row():
            # 左侧：输入和参数
            with gr.Column(scale=1):
                gr.Markdown("### 📁 输入视频")
                input_video = gr.File(
                    label="上传视频文件",
                    file_types=['video'],
                    type="filepath"
                )
                
                # 添加输入视频预览
                input_video_preview = gr.Video(
                    label="输入视频预览",
                    height=300
                )
                
                gr.Markdown("### ⚙️ 推理参数")
                with gr.Accordion("高级设置", open=False):
                    num_inference_steps = gr.Slider(
                        minimum=1, maximum=100, value=50, step=1,
                        label="推理步数 (越高质量越好，但耗时更长)"
                    )
                    guidance_scale = gr.Slider(
                        minimum=1, maximum=20, value=6, step=1,
                        label="引导强度"
                    )
                    tile_size = gr.Slider(
                        minimum=64, maximum=256, value=128, step=32,
                        label="瓦片大小 (影响内存使用)"
                    )
                    upscale = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                        label="缩放倍数 (0为自动缩放到1024)"
                    )
                    use_dynamic_cfg = gr.Checkbox(
                        label="使用动态CFG", value=False
                    )
                    seed = gr.Number(
                        label="随机种子", value=42, precision=0
                    )
                
                process_btn = gr.Button("🎯 开始增强", variant="primary", size="lg")
            
            # 右侧：输出和预览
            with gr.Column(scale=1):
                gr.Markdown("### 📺 输出结果")
                output_video = gr.Video(
                    label="增强后的视频",
                    height=400
                )
                
                output_info = gr.Textbox(
                    label="处理信息",
                    lines=6,
                    value="等待处理...",
                    interactive=False
                )
        
        # 示例和帮助
        with gr.Accordion("💡 使用提示", open=False):
            gr.Markdown("""
            **参数说明：**
            - **推理步数**: 控制生成质量，推荐值：快速测试用10-20，高质量用50-100
            - **引导强度**: 控制对原视频的保真度，通常6-8效果较好
            - **瓦片大小**: 影响显存使用，较小的值节省显存但可能影响质量
            - **缩放倍数**: 0表示自动缩放短边到1024像素，其他值为直接缩放倍数
            
            **支持格式**: MP4, AVI, MOV, MKV等常见视频格式
            
            **硬件要求**: 推荐使用16GB+显存的GPU，较大视频可能需要调小瓦片大小
            """)
        
        # 事件绑定
        # 视频上传预览
        input_video.change(
            fn=update_video_preview,
            inputs=input_video,
            outputs=input_video_preview
        )
        
        init_btn.click(
            fn=model.initialize_models,
            outputs=init_status,
            show_progress=True
        )
        
        process_btn.click(
            fn=model.process_video,
            inputs=[
                input_video, num_inference_steps, guidance_scale, 
                tile_size, upscale, use_dynamic_cfg, seed
            ],
            outputs=[output_video, output_info],
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
