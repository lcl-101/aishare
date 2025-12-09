#!/usr/bin/env python3
"""
HunyuanVideo-Avatar 独立演示应用
运行方式: python app.py
"""

import os
import sys
import math
import uuid
import warnings
import datetime
import numpy as np
import torch
import imageio
import gradio as gr
from PIL import Image
from einops import rearrange
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

# 设置环境变量
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["PYTHONPATH"] = "./"
os.environ["MODEL_BASE"] = "./weights"  # 设置模型基础路径
os.environ["DISABLE_SP"] = "1"  # 禁用序列并行，单GPU模式
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
warnings.filterwarnings("ignore")

# 配置
MODEL_BASE = "./weights"
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ===================== 模型加载 =====================

def load_models(checkpoint_path, use_fp8=False, cpu_offload=False):
    """加载所有需要的模型"""
    import torch.distributed as dist
    from hymm_sp.config import parse_args
    from hymm_sp.sample_inference_audio import HunyuanVideoSampler
    from transformers import WhisperModel, AutoFeatureExtractor
    from hymm_sp.data_kits.face_align import AlignImage
    from hymm_sp.modules.parallel_states import initialize_sequence_parallel_state, nccl_info
    
    # 初始化分布式环境（单GPU模式）
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0
        )
    
    # 初始化序列并行状态
    initialize_sequence_parallel_state(1)
    
    # 构建参数
    sys.argv = [
        'app.py',
        '--input', 'assets/test.csv',
        '--ckpt', checkpoint_path,
        '--sample-n-frames', '129',
        '--seed', '128',
        '--image-size', '704',
        '--cfg-scale', '7.5',
        '--infer-steps', '50',
        '--use-deepcache', '1',
        '--flow-shift-eval-video', '5.0',
    ]
    
    if use_fp8:
        sys.argv.append('--use-fp8')
    if cpu_offload:
        sys.argv.append('--cpu-offload')
    
    args = parse_args()
    
    print("=" * 60)
    print("Loading HunyuanVideo-Avatar models...")
    print("=" * 60)
    
    # 加载主模型
    hunyuan_sampler = HunyuanVideoSampler.from_pretrained(checkpoint_path, args=args)
    args = hunyuan_sampler.args
    device = torch.device("cuda")
    
    # 加载 Whisper 音频特征提取器
    print("Loading Whisper model...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"{MODEL_BASE}/ckpts/whisper-tiny/")
    wav2vec = WhisperModel.from_pretrained(f"{MODEL_BASE}/ckpts/whisper-tiny/").to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)
    
    # 加载人脸对齐模型
    print("Loading face alignment model...")
    det_path = os.path.join(MODEL_BASE, 'ckpts/det_align/detface.pt')
    align_instance = AlignImage("cuda", det_path=det_path)
    
    print("=" * 60)
    print("All models loaded successfully!")
    print("=" * 60)
    
    return {
        'sampler': hunyuan_sampler,
        'args': args,
        'wav2vec': wav2vec,
        'feature_extractor': feature_extractor,
        'align_instance': align_instance,
        'device': device,
    }


# ===================== 数据预处理 =====================

def preprocess_data(args, image_path, audio_path, prompt, feature_extractor):
    """预处理输入数据"""
    from hymm_sp.data_kits.audio_dataset import get_audio_feature
    
    llava_transform = transforms.Compose([
        transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # 处理 prompt
    if prompt is None or prompt.strip() == "":
        prompt = "Authentic, Realistic, Natural, High-quality, Lens-Fixed."
    else:
        prompt = "Authentic, Realistic, Natural, High-quality, Lens-Fixed, " + prompt
    
    fps = 25
    img_size = args.image_size
    
    # 处理参考图像
    ref_image = Image.open(image_path).convert('RGB')
    w, h = ref_image.size
    scale = img_size / min(w, h)
    new_w = round(w * scale / 64) * 64
    new_h = round(h * scale / 64) * 64
    
    if img_size == 704:
        img_size_long = 1216
        if new_w * new_h > img_size * img_size_long:
            scale = math.sqrt(img_size * img_size_long / w / h)
            new_w = round(w * scale / 64) * 64
            new_h = round(h * scale / 64) * 64
    
    ref_image = ref_image.resize((new_w, new_h), Image.LANCZOS)
    ref_image = torch.from_numpy(np.array(ref_image))
    
    # 处理音频
    audio_input, audio_len = get_audio_feature(feature_extractor, audio_path)
    audio_prompts = audio_input[0]
    
    # 运动参数
    motion_bucket_id_heads = torch.from_numpy(np.array([25] * 4))
    motion_bucket_id_exps = torch.from_numpy(np.array([30] * 4))
    fps = torch.from_numpy(np.array(fps))
    
    # 处理参考图像用于VAE和LLaVA
    to_pil = ToPILImage()
    pixel_value_ref = rearrange(ref_image.clone().unsqueeze(0), "b h w c -> b c h w")
    pixel_value_ref_llava = [llava_transform(to_pil(image)) for image in pixel_value_ref]
    pixel_value_ref_llava = torch.stack(pixel_value_ref_llava, dim=0)
    
    batch = {
        "text_prompt": [prompt],
        "audio_path": [audio_path],
        "image_path": [image_path],
        "fps": fps.unsqueeze(0).to(dtype=torch.float16),
        "audio_prompts": audio_prompts.unsqueeze(0).to(dtype=torch.float16),
        "audio_len": [audio_len],
        "motion_bucket_id_exps": motion_bucket_id_exps.unsqueeze(0),
        "motion_bucket_id_heads": motion_bucket_id_heads.unsqueeze(0),
        "pixel_value_ref": pixel_value_ref.unsqueeze(0).to(dtype=torch.float16),
        "pixel_value_ref_llava": pixel_value_ref_llava.unsqueeze(0).to(dtype=torch.float16)
    }
    
    return batch, audio_len


# ===================== 推理函数 =====================

def generate_video(audio_path, image, prompt, models, progress=gr.Progress()):
    """生成视频的主函数"""
    if image is None:
        raise gr.Error("请上传参考图像")
    if audio_path is None:
        raise gr.Error("请上传音频文件")
    
    progress(0.1, desc="准备数据...")
    
    # 保存图像到临时文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_image_path = os.path.join(TEMP_DIR, f"input_{timestamp}.png")
    
    # image 是 numpy array (H, W, C) RGB格式
    Image.fromarray(image).save(temp_image_path)
    
    try:
        progress(0.2, desc="预处理数据...")
        
        # 预处理
        batch, audio_len = preprocess_data(
            models['args'],
            temp_image_path,
            audio_path,
            prompt,
            models['feature_extractor']
        )
        
        progress(0.3, desc="生成视频中 (这可能需要几分钟)...")
        
        # 推理
        outputs = models['sampler'].predict(
            models['args'],
            batch,
            models['wav2vec'],
            models['feature_extractor'],
            models['align_instance']
        )
        
        if outputs is None:
            raise gr.Error("视频生成失败")
        
        progress(0.9, desc="保存视频...")
        
        # 处理输出
        samples = outputs["samples"]
        sample = samples[0].unsqueeze(0)
        sample = sample[:, :, :audio_len]
        
        video = sample[0].permute(1, 2, 3, 0).clamp(0, 1).cpu().numpy()
        video = (video * 255.).astype(np.uint8)
        
        # 保存视频
        output_video_path = os.path.join(TEMP_DIR, f"output_{timestamp}.mp4")
        imageio.mimsave(output_video_path, video, fps=25)
        
        # 添加音频
        output_with_audio = output_video_path.replace(".mp4", "_audio.mp4")
        os.system(f"ffmpeg -i '{output_video_path}' -i '{audio_path}' -shortest '{output_with_audio}' -y -loglevel quiet")
        
        if os.path.exists(output_with_audio):
            os.remove(output_video_path)
            output_video_path = output_with_audio
        
        progress(1.0, desc="完成!")
        
        return output_video_path
        
    finally:
        # 清理临时图像文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# ===================== Gradio 界面 =====================

def create_demo(models):
    """创建 Gradio 界面"""
    
    # 从 test.csv 中选取的示例
    examples = [
        # [audio, image, prompt]
        ["assets/audio/2.WAV", "assets/image/1.png", "A person sits cross-legged by a campfire in a forested area."],
        ["assets/audio/2.WAV", "assets/image/2.png", "A person with long blonde hair wearing a green jacket, standing in a forested area during twilight."],
        ["assets/audio/3.WAV", "assets/image/3.png", "A person playing guitar by a campfire in a forest."],
        ["assets/audio/3.WAV", "assets/image/4.png", "A person wearing a green jacket stands in a forested area, with sunlight filtering through the trees."],
        ["assets/audio/4.WAV", "assets/image/src1.png", "A person sits cross-legged by a campfire in a forest at dusk."],
        ["assets/audio/4.WAV", "assets/image/src2.png", "A person in a green jacket stands in a forest at dusk."],
    ]
    
    def run_generation(audio_path, image, prompt):
        return generate_video(audio_path, image, prompt, models)
    
    with gr.Blocks(title="HunyuanVideo-Avatar Demo") as demo:
        gr.Markdown("""
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">
                🎬 Tencent HunyuanVideo-Avatar Demo
            </h1>
            <p style="color: #888;">上传一张人像图片和一段音频，生成说话视频</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt (提示词)", 
                    value="a person is speaking.",
                    placeholder="描述视频内容，例如: a man is speaking, a woman is talking..."
                )
                
                audio_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="🎵 上传音频 (Upload Audio)",
                )
                
                image_input = gr.Image(
                    label="🖼️ 上传参考图像 (Reference Image)",
                    type="numpy",
                    height=400
                )
                
                generate_btn = gr.Button("🚀 生成视频 (Generate)", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_video = gr.Video(label="🎬 生成的视频 (Generated Video)")
        
        # 添加示例
        gr.Markdown("### 📌 示例 (Examples) - 点击使用")
        gr.Examples(
            examples=examples,
            inputs=[audio_input, image_input, prompt],
            label="",
        )
        
        # 使用说明
        gr.Markdown("""
        ---
        ### 📖 使用说明
        1. **上传音频**: 支持 WAV 格式的语音文件
        2. **上传图像**: 上传一张清晰的人像照片（正面照效果最佳）
        3. **设置提示词**: 描述视频内容（可选）
        4. **点击生成**: 等待几分钟即可获得说话视频
        
        ⚠️ **注意**: 首次生成可能需要较长时间，请耐心等待
        """)
        
        generate_btn.click(
            fn=run_generation,
            inputs=[audio_input, image_input, prompt],
            outputs=[output_video],
        )
    
    return demo


# ===================== 主程序 =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HunyuanVideo-Avatar Demo")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    parser.add_argument("--use-fp8", action="store_true", help="使用 FP8 量化（节省显存）")
    parser.add_argument("--cpu-offload", action="store_true", help="CPU offload（低显存模式）")
    cmd_args = parser.parse_args()
    
    # 确定模型路径
    if cmd_args.use_fp8:
        checkpoint_path = f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
    else:
        checkpoint_path = f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Use FP8: {cmd_args.use_fp8}")
    print(f"CPU Offload: {cmd_args.cpu_offload}")
    
    # 加载模型
    models = load_models(checkpoint_path, cmd_args.use_fp8, cmd_args.cpu_offload)
    
    # 创建并启动 Gradio
    demo = create_demo(models)
    demo.launch(
        server_name="0.0.0.0",
        server_port=cmd_args.port,
        share=cmd_args.share,
        allowed_paths=["/"]
    )
