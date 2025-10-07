# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import tempfile
import zipfile

warnings.filterwarnings('ignore')

import random
import numpy as np
from PIL import Image, ImageDraw
import imageio
import cv2
import torchvision
import gradio as gr

import torch
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool


class WanAlphaWebUI:
    def __init__(self, auto_load_model=False):
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.default_prompt = "This video has a transparent background. Close-up shot. A colorful parrot is flying. Realistic style."
        
        # Model paths
        self.ckpt_dir = "./checkpoints/Wan2.1-T2V-14B"
        self.vae_lora_checkpoint = "./checkpoints/Wan-Alpha/decoder.bin"
        self.lora_path = "./checkpoints/Wan-Alpha/epoch-13-1500.safetensors"
        self.lightx2v_path = "./checkpoints/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
        
        # Auto load model if requested
        if auto_load_model:
            print("🚀 自动加载模型...")
            result = self.load_model()
            if "Error" in result:
                print(f"❌ {result}")
                sys.exit(1)
            else:
                print("✅ 模型加载成功！")
    
    def load_model(self):
        """Load the Wan-Alpha model"""
        if self.model is not None:
            return "Model already loaded!"
        
        try:
            cfg = WAN_CONFIGS['t2v-14B']
            logging.info("Creating Wan-Alpha pipeline.")
            
            self.model = wan.WanT2V_dora_lightx2v(
                config=cfg,
                checkpoint_dir=self.ckpt_dir,
                vae_lora_checkpoint=self.vae_lora_checkpoint,
                lora_path=self.lora_path,
                lightx2v_path=self.lightx2v_path,
                lora_ratio=1.0,
                device_id=0,
                rank=0,
                t5_fsdp=False,  # Disabled for single GPU
                dit_fsdp=False,  # Disabled for single GPU
                use_usp=False,  # Disabled for single GPU
                t5_cpu=False,
            )
            return "Model loaded successfully!"
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def render_video(self, tensor_fgr, tensor_pha, nrow=8, normalize=True, value_range=(-1, 1)):
        """Render video from foreground and alpha tensors"""
        tensor_fgr = tensor_fgr.clamp(min(value_range), max(value_range))
        tensor_fgr = torch.stack([
            torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor_fgr.unbind(2)
        ], dim=1).permute(1, 2, 3, 0)
        tensor_fgr = (tensor_fgr * 255).type(torch.uint8).cpu()

        tensor_pha = tensor_pha.clamp(min(value_range), max(value_range))
        tensor_pha = torch.stack([
            torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor_pha.unbind(2)
        ], dim=1).permute(1, 2, 3, 0)
        tensor_pha = (tensor_pha * 255).type(torch.uint8).cpu()

        frames = []
        frames_fgr = []
        frames_pha = []
        for frame_fgr, frame_pha in zip(tensor_fgr.numpy(), tensor_pha.numpy()):
            frame_pha = (0.0 + frame_pha[:,:,0:1] + frame_pha[:,:,1:2] + frame_pha[:,:,2:3]) / 3.
            frame = np.concatenate([frame_fgr[:,:,::-1], frame_pha.astype(np.uint8)], axis=2)
            frames.append(frame)
            frames_fgr.append(frame_fgr)
            frames_pha.append(frame_pha)

        def create_checkerboard(size=30, pattern_size=(830, 480), color1=(140, 140, 140), color2=(113, 113, 113)):
            img = Image.new('RGB', (pattern_size[0], pattern_size[1]), color1)
            draw = ImageDraw.Draw(img)
            for i in range(0, pattern_size[0], size):
                for j in range(0, pattern_size[1], size):
                    if (i + j) // size % 2 == 0:
                        draw.rectangle([i, j, i+size, j+size], fill=color2)
            return img

        def blender_background(frame_rgba, checkerboard):
            alpha_channel = frame_rgba[:, :, 3:] / 255. 
            checkerboard = np.array(checkerboard)
            checkerboard = cv2.resize(checkerboard, (frame_rgba.shape[1], frame_rgba.shape[0]))

            frame_rgb = frame_rgba[:, :, :3] * alpha_channel + checkerboard * (1-alpha_channel)
            return frame_rgb.astype(np.uint8)[:,:,::-1]
        
        checkerboard = create_checkerboard()
        video_checkerboard = [blender_background(f, checkerboard) for f in frames]
        
        # Create alpha video (grayscale visualization of alpha channel)
        video_alpha = []
        for frame_pha in frames_pha:
            # Convert alpha to RGB grayscale (0-255)
            alpha_frame = np.stack([frame_pha, frame_pha, frame_pha], axis=2).astype(np.uint8)
            video_alpha.append(alpha_frame[:,:,:,0])  # Remove extra dimension

        return video_checkerboard, frames, video_alpha

    def save_video(self, save_path, target_frames, fps=16):
        """Save video frames to file"""
        writer = imageio.get_writer(
            save_path, fps=fps, codec='libx264', quality=8)
        for frame in target_frames:
            writer.append_data(frame)
        writer.close()

    def generate_video(self, prompt, size, frame_num, sample_steps, guide_scale, seed, negative_prompt, progress=gr.Progress()):
        """Generate video from text prompt"""
        if self.model is None:
            return None, None, "请先加载模型！"
        
        if not prompt.strip():
            prompt = self.default_prompt
        
        try:
            progress(0.1, desc="开始生成...")
            
            # Set seed
            if seed == -1:
                seed = random.randint(0, sys.maxsize)
            
            logging.info(f"Generating video with prompt: {prompt}")
            progress(0.3, desc="生成视频中...")
            
            videos = self.model.generate(
                prompt,
                size=SIZE_CONFIGS[size],
                frame_num=frame_num,
                shift=5.0,  # Default sample_shift
                sample_solver='unipc',
                sampling_steps=sample_steps,
                guide_scale=guide_scale,
                seed=seed,
                n_prompt=negative_prompt,
                offload_model=True  # Enable offloading for single GPU
            )
            
            progress(0.8, desc="渲染视频中...")
            
            video_fgr, video_pha = videos
            video_checkerboard, frames, video_alpha = self.render_video(video_fgr[None], video_pha[None])
            
            # Create temporary files
            temp_dir = tempfile.mkdtemp()
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save checkerboard video (preview with background)
            video_path = os.path.join(temp_dir, f"preview_{formatted_time}.mp4")
            self.save_video(video_path, video_checkerboard)
            
            # Save alpha video (grayscale alpha channel)
            alpha_video_path = os.path.join(temp_dir, f"alpha_{formatted_time}.mp4")
            self.save_video(alpha_video_path, video_alpha)
            
            # Save PNG frames as zip
            zip_path = os.path.join(temp_dir, f"frames_{formatted_time}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for idx, img in enumerate(frames):
                    success, buffer = cv2.imencode(".png", img)
                    if success:
                        filename = f"img_{idx:03d}.png"
                        zipf.writestr(filename, buffer.tobytes())
            
            progress(1.0, desc="完成！")
            
            return video_path, alpha_video_path, zip_path, f"视频生成成功！使用的随机种子: {seed}"
            
        except Exception as e:
            error_msg = f"生成视频时出错: {str(e)}"
            logging.error(error_msg)
            return None, None, None, error_msg

    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="Wan-Alpha 视频生成", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎬 Wan-Alpha 视频生成 WebUI")
            gr.Markdown("使用 Wan-Alpha 模型从文本提示生成透明背景视频。")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input parameters
                    gr.Markdown("## 生成参数")
                    prompt = gr.Textbox(
                        label="提示词", 
                        value=self.default_prompt,
                        placeholder="在此输入您的提示词...",
                        lines=3
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="负面提示词",
                        value="色调艳丽，过曝，静态，细节模糊不清，字幕，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        lines=2
                    )
                    
                    with gr.Row():
                        size = gr.Dropdown(
                            choices=["832*480", "768*512", "1024*576"],
                            value="832*480",
                            label="视频尺寸"
                        )
                        frame_num = gr.Slider(
                            minimum=17, maximum=161, value=81, step=16,
                            label="帧数 (4n+1)"
                        )
                    
                    with gr.Row():
                        sample_steps = gr.Slider(
                            minimum=1, maximum=50, value=4,
                            label="采样步数"
                        )
                        guide_scale = gr.Slider(
                            minimum=0.1, maximum=10.0, value=1.0, step=0.1,
                            label="引导系数"
                        )
                    
                    seed = gr.Number(
                        label="随机种子 (-1为随机)", 
                        value=-1, 
                        precision=0
                    )
                    
                    generate_btn = gr.Button("🎥 生成视频", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("## 输出结果")
                    result_status = gr.Textbox(label="生成状态", interactive=False)
                    
                    with gr.Row():
                        output_video = gr.Video(
                            label="预览视频 (Preview Video)",
                            height=350
                        )
                        alpha_video = gr.Video(
                            label="Alpha 视频 (Alpha Video)",
                            height=350
                        )
                    
                    output_frames = gr.File(
                        label="下载PNG帧序列 (ZIP)",
                        file_count="single"
                    )
            
            # Examples section
            gr.Markdown("## 📝 示例提示词")
            gr.Examples(
                examples=[
                    ["Medium shot. A little girl holds a bubble wand and blows out colorful bubbles that float and pop in the air. The background of this video is transparent. Realistic style."],
                    ["Close-up. A coffee bean falls into a cup of hot water, and the brown liquid slowly spreads out. The background of this video is transparent. Realistic style."],
                    ["Close-up. A bird takes off from a branch, its wings fluttering and lifting tiny feathers. The background of this video is transparent. Realistic style."],
                    ["Close-up. A candle burns, its flame swaying with the air, and wax slowly drips. The background of this video is transparent. Realistic style."],
                    ["This video has a transparent background. A young boy, wearing tracksuits, drinks water. He has a relaxed expression, holding a clear plastic cup and looking forward. Realistic style. Medium shot. Slight upward angle."],
                    ["This video has a transparent background. An attractive Asian woman, dressed in a stylish urban business suit, smiles as a gentle breeze rustles her long, flowing black hair, sending strands flying. Her eyes are gentle and vibrant, and a contented smile plays on her lips. Realistic style. Medium shot. Slight upward angle."],
                    ["This video has a transparent background. A man in sportswear drinks water, his expression relaxed, his eyes looking forward. Realistic style. Medium shot."],
                    ["The video has a transparent background. A woman in a gorgeous gown holds a dimly lit candle, its flame framing her dreamlike silhouette. Her expression is serene, as if she's lost in thought. The video is shot in a romantic, retro style, with a close-up of her body."],
                    ["This video has a transparent background. Realistic style. A woman in casual home clothes elegantly holds a clear glass filled with orange juice. She has warm eyes and soft brown curls. Medium shot."],
                    ["This video has a transparent background. The edges of the crystal flower's petals shimmer with a cool blue-white light, while the center is faintly luminous. Realistic style. Medium shot."],
                    ["This video has a transparent background. A silvery-white halo flickers in a circular pattern, spreading evenly. Realistic style. Medium shot."],
                    ["This video has a transparent background. A street lamp shade glows orange-red, and the wick flickers. Realistic style. Medium shot."],
                ],
                inputs=[prompt],
                label="点击下方示例自动填充提示词",
                examples_per_page=6
            )
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_video,
                inputs=[prompt, size, frame_num, sample_steps, guide_scale, seed, negative_prompt],
                outputs=[output_video, alpha_video, output_frames, result_status]
            )
            

        
        return demo


def main():
    print("🎬 Wan-Alpha 视频生成 WebUI")
    print("=" * 50)
    print("📁 检查模型文件...")
    
    # Create webui instance (without auto-loading first)
    temp_webui = WanAlphaWebUI(auto_load_model=False)
    
    # Check required files
    required_files = [
        (temp_webui.ckpt_dir, "主模型检查点目录"),
        (temp_webui.vae_lora_checkpoint, "VAE LoRA 检查点"),
        (temp_webui.lora_path, "LoRA 检查点"),
        (temp_webui.lightx2v_path, "LightX2V 检查点"),
    ]
    
    missing_files = []
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            missing_files.append(f"❌ {description}: {file_path}")
        else:
            print(f"✅ {description}: 已找到")
    
    if missing_files:
        print("\n⚠️  缺少必需文件:")
        for missing in missing_files:
            print(missing)
        print("\n请确保所有模型文件都在正确的位置。")
        return
    
    print("✅ 所有必需的模型文件都已找到！")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 检测到 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  警告: CUDA 不可用，将使用 CPU (会很慢)")
    
    print("=" * 50)
    
    # Create webui instance with auto-loading
    webui = WanAlphaWebUI(auto_load_model=True)
    
    print("=" * 50)
    print("🌐 启动 WebUI 服务器...")
    print("📝 在浏览器中打开: http://localhost:7860")
    print("🔗 远程访问请使用: http://YOUR_SERVER_IP:7860")
    print("=" * 50)
    
    # Create and launch interface
    demo = webui.create_interface()
    
    # Launch with public access if needed
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want public sharing
        debug=False,
        show_error=True
    )


if __name__ == "__main__":
    main()