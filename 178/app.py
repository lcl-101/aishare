# -*- coding: utf-8 -*-
# ==============================================================================
# arxive: https://arxiv.org/abs/2507.03905
# GitHUb: https://github.com/antgroup/echomimic_v3
# Project Page: https://antgroup.github.io/ai/echomimic_v3/
# ==============================================================================

import os
import math
import datetime
from functools import partial
import glob

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from moviepy import VideoFileClip, AudioFileClip
import librosa

# Custom modules
from diffusers import FlowMatchEulerDiscreteScheduler

from src.dist import set_multi_gpus_devices
from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import  CLIPModel
from src.wan_text_encoder import  WanT5EncoderModel
from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline

from src.utils import (
    filter_kwargs,
    get_image_to_video_latent3,
    save_videos_grid,
)
from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.cache_utils import get_teacache_coefficients

from src.face_detect import get_mask_coord

import argparse
import gradio as gr
import random
import gc

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="0.0.0.0", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7860, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
args = parser.parse_args()


# --------------------- Configuration ---------------------
class Config:
    def __init__(self):
        # General settings
        self.ulysses_degree = 1
        self.ring_degree = 1
        self.fsdp_dit = False

        # Pipeline parameters
        self.num_skip_start_steps = 5
        self.teacache_offload = False  # 改为与官方一致
        self.cfg_skip_ratio = 0
        self.enable_riflex = False
        self.riflex_k = 6

        # Paths
        self.config_path = "config/config.yaml"
        self.model_name = "models/Wan2.1-Fun-V1.1-1.3B-InP"
        self.transformer_path = "models/transformer/diffusion_pytorch_model.safetensors"

        # Sampler and audio settings
        self.sampler_name = "Flow_DPM++"
        self.audio_scale = 1.0
        self.enable_teacache = False  # 改为与官方一致
        self.teacache_threshold = 0.1
        self.shift = 5.0
        self.use_un_ip_mask = False

        self.partial_video_length = 49
        self.overlap_video_length = 8
        self.neg_scale = 1.5
        self.neg_steps = 2
        self.guidance_scale = 4.0  # 改为与官方一致 4.0 ~ 6.0
        self.audio_guidance_scale = 2.5 #2.0 ~ 3.0
        self.use_dynamic_cfg = True
        self.use_dynamic_acfg = True
        self.seed = 43
        self.num_inference_steps = 20
        self.lora_weight = 1.0

        # Model settings
        self.weight_dtype = torch.bfloat16
        self.sample_size = [768, 768]
        self.fps = 25

        # Test data paths
        self.wav2vec_model_dir = "models/wav2vec2-base-960h"


# --------------------- Helper Functions ---------------------
def load_wav2vec_models(wav2vec_model_dir):
    """Load Wav2Vec models for audio feature extraction."""
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
    # 与官方保持一致，不移动到GPU
    model = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).eval()
    model.requires_grad_(False)
    return processor, model


def extract_audio_features(audio_path, processor, model):
    """Extract audio features using Wav2Vec."""
    sr = 16000
    audio_segment, sample_rate = librosa.load(audio_path, sr=sr)
    input_values = processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt").input_values
    # 与官方保持一致，根据模型设备动态处理
    if hasattr(model, 'device'):
        input_values = input_values.to(model.device)
    features = model(input_values).last_hidden_state
    return features.squeeze(0)


def get_sample_size(image, default_size):
    """Calculate the sample size based on the input image dimensions."""
    width, height = image.size
    original_area = width * height
    default_area = default_size[0] * default_size[1]

    if default_area < original_area:
        ratio = math.sqrt(original_area / default_area)
        width = width / ratio // 16 * 16
        height = height / ratio // 16 * 16
    else:
        width = width // 16 * 16
        height = height // 16 * 16

    return int(height), int(width)


def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    
    mask = mask.reshape(-1)
    return mask.float()


def get_examples():
    """获取演示样例，基于 datasets 目录中的文件"""
    examples = []
    datasets_path = "datasets/echomimicv3_demos"
    
    # 检查目录是否存在
    if not os.path.exists(datasets_path):
        print(f"Warning: Demo datasets directory not found: {datasets_path}")
        return examples
    
    try:
        # 获取所有音频文件，支持多种扩展名
        audio_extensions = ["*.WAV", "*.wav"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(datasets_path, "audios", ext)))
        
        # 过滤掉非音频文件，确保只处理真正的音频文件
        valid_audio_files = []
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            # 排除一些明显不是音频的文件
            if not filename.lower().endswith(('.wav', '.WAV')):
                continue
            # 确保文件确实存在且不是目录
            if os.path.isfile(audio_file):
                valid_audio_files.append(audio_file)
        
        # 加载所有有效的示例
        audio_files = sorted(set(valid_audio_files))
        print(f"Found {len(audio_files)} audio files for examples")
        
        for audio_file in audio_files:
            # 提取文件名（不包含扩展名）
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            
            # 查找对应的图片文件，使用官方的扩展名顺序
            img_extensions = ['png', 'jpeg', 'jpg']  # 与官方保持一致
            img_file = None
            for ext in img_extensions:
                potential_img = os.path.join(datasets_path, "imgs", f"{base_name}.{ext}")
                if os.path.exists(potential_img):
                    img_file = potential_img
                    break
            
            # 查找对应的 prompt 文件
            prompt_file = os.path.join(datasets_path, "prompts", f"{base_name}.txt")
            prompt_text = ""
            if os.path.exists(prompt_file):
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        # 限制提示词长度，避免过长
                        if len(content) > 500:
                            content = content[:500] + "..."
                        prompt_text = content
                except Exception as e:
                    print(f"Warning: Failed to read prompt file {prompt_file}: {e}")
                    prompt_text = ""
            
            # 如果找到了对应的图片和音频文件，添加到示例中
            if img_file and os.path.exists(img_file):
                examples.append([
                    img_file,  # image
                    audio_file,  # audio
                    prompt_text,  # prompt
                    "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. 手部快速摆动, 手指频繁抽搐, 夸张手势, 重复机械性动作.",  # negative_prompt
                    -1  # seed_param
                ])
                print(f"Added example: {base_name}")
            else:
                print(f"Skipped {base_name}: missing image file")
                
    except Exception as e:
        print(f"Error loading demo examples: {e}")
        return []
    
    return examples


# Initialize configuration
config = Config()

# Set up multi-GPU devices
device = set_multi_gpus_devices(config.ulysses_degree, config.ring_degree)

# Load configuration file
cfg = OmegaConf.load(config.config_path)

# Load models
transformer = WanTransformerAudioMask3DModel.from_pretrained(
    os.path.join(config.model_name, cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
    torch_dtype=config.weight_dtype,
)
if config.transformer_path is not None:
    if config.transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(config.transformer_path)
    else:
        state_dict = torch.load(config.transformer_path)
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(config.model_name, cfg['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
).to(config.weight_dtype)

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
    torch_dtype=config.weight_dtype,
).eval()

clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(config.model_name, cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(config.weight_dtype).eval()

# Load scheduler
scheduler_cls = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[config.sampler_name]
scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(cfg['scheduler_kwargs'])))

# Create pipeline
pipeline = WanFunInpaintAudioPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
)
pipeline.to(device=device)

# Enable TeaCache if required
if config.enable_teacache:
    coefficients = get_teacache_coefficients(config.model_name)
    pipeline.transformer.enable_teacache(
        coefficients, config.num_inference_steps, config.teacache_threshold,
        num_skip_start_steps=config.num_skip_start_steps, offload=config.teacache_offload
    )# Load Wav2Vec models
print("Loading Wav2Vec models...")
wav2vec_processor, wav2vec_model = load_wav2vec_models(config.wav2vec_model_dir)

# 预先生成示例列表，避免在界面中重复调用
print("Loading demo examples...")
DEMO_EXAMPLES = get_examples()
print(f"✅ Loaded {len(DEMO_EXAMPLES)} demo examples")

if len(DEMO_EXAMPLES) == 0:
    print("⚠️  No demo examples found. Please check the datasets directory structure.")


def generate(
    image,
    audio,
    prompt,
    negative_prompt,
    seed_param,
    # 添加高级参数
    audio_guidance_scale,
    guidance_scale,
    num_inference_steps,
    teacache_threshold,
    partial_video_length,
    enable_teacache,
    progress=gr.Progress()
):
    # 输入验证
    if image is None:
        raise gr.Error("请上传参考图片！")
    if audio is None:
        raise gr.Error("请上传音频文件！")
    
    progress(0, desc="初始化...")
    
    # 动态配置TeaCache
    if enable_teacache and not config.enable_teacache:
        # 如果用户启用了TeaCache但默认配置关闭，需要动态启用
        coefficients = get_teacache_coefficients(config.model_name)
        pipeline.transformer.enable_teacache(
            coefficients, int(num_inference_steps), float(teacache_threshold),
            num_skip_start_steps=config.num_skip_start_steps, offload=config.teacache_offload
        )
    elif not enable_teacache and config.enable_teacache:
        # 如果用户关闭了TeaCache但默认配置启用，需要动态关闭
        if hasattr(pipeline.transformer, 'disable_teacache'):
            pipeline.transformer.disable_teacache()
    
    if seed_param<0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "outputs"
    os.makedirs(save_path, exist_ok=True)

    progress(0.1, desc="设置随机种子...")
    # Process test cases
    generator = torch.Generator(device=device).manual_seed(seed)

    progress(0.2, desc="处理参考图片...")
    # Load reference image and prompt
    ref_img = Image.open(image).convert("RGB")

    y1, y2, x1, x2, h_, w_ = get_mask_coord(image)

    progress(0.3, desc="提取音频特征...")
    # Extract audio features - 验证音频文件存在性
    if not os.path.exists(audio):
        # 尝试不同的扩展名，与官方逻辑一致
        audio_alt = audio.replace("WAV", "wav") if "WAV" in audio else audio.replace("wav", "WAV")
        if os.path.exists(audio_alt):
            audio = audio_alt
        else:
            raise gr.Error(f"音频文件不存在: {audio}")
    
    audio_clip = AudioFileClip(audio)
    audio_features = extract_audio_features(audio, wav2vec_processor, wav2vec_model)
    audio_embeds = audio_features.unsqueeze(0).to(device=device, dtype=config.weight_dtype)

    progress(0.4, desc="计算视频参数...")
    # Calculate video length and latent frames
    video_length = int(audio_clip.duration * config.fps)
    video_length = (
        int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
        if video_length != 1 else 1
    )
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
    
    print(f"视频总长度: {video_length} 帧 ({audio_clip.duration:.2f}秒)")

    if config.enable_riflex:
        pipeline.transformer.enable_riflex(k = config.riflex_k, L_test = latent_frames)

    progress(0.5, desc="准备图像处理...")
    # Adjust sample size and create IP mask
    sample_height, sample_width = get_sample_size(ref_img, config.sample_size)
    downratio = math.sqrt(sample_height * sample_width / h_ / w_)
    coords = (
        y1 * downratio // 16, y2 * downratio // 16,
        x1 * downratio // 16, x2 * downratio // 16,
        sample_height // 16, sample_width // 16,
    )
    ip_mask = get_ip_mask(coords).unsqueeze(0)
    ip_mask = torch.cat([ip_mask]*3).to(device=device, dtype=config.weight_dtype)

    # 恢复使用官方配置
    # 使用用户配置的片段长度
    user_partial_video_length = int(partial_video_length)
    partial_video_length_adjusted = int((user_partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (partial_video_length_adjusted - 1) // vae.config.temporal_compression_ratio + 1

    # get clip image
    _, _, clip_image = get_image_to_video_latent3(ref_img, None, video_length=partial_video_length_adjusted, sample_size=[sample_height, sample_width])

    progress(0.6, desc="开始生成视频...")
    # Generate video in chunks
    init_frames = 0
    last_frames = init_frames + partial_video_length_adjusted
    new_sample = None
    
    current_chunk = 0

    while init_frames < video_length:
        current_chunk += 1
        # 基于剩余帧数动态计算进度
        progress_ratio = min(0.9, 0.6 + 0.3 * (init_frames / video_length))
        remaining_frames = video_length - init_frames
        progress(progress_ratio, desc=f"处理视频片段 {current_chunk} (剩余约 {remaining_frames/config.fps:.1f}秒)...")
        
        if last_frames >= video_length:
            partial_video_length_adjusted = video_length - init_frames
            partial_video_length_adjusted = (
                int((partial_video_length_adjusted - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
                if video_length != 1 else 1
            )
            latent_frames = (partial_video_length_adjusted - 1) // vae.config.temporal_compression_ratio + 1

            if partial_video_length_adjusted <= 0:
                break

        input_video, input_video_mask, _ = get_image_to_video_latent3(
            ref_img, None, video_length=partial_video_length_adjusted, sample_size=[sample_height, sample_width]
        )

        partial_audio_embeds = audio_embeds[:, init_frames * 2 : (init_frames + partial_video_length_adjusted) * 2]

        sample = pipeline(
            prompt,
            num_frames            = partial_video_length_adjusted,
            negative_prompt       = negative_prompt,
            audio_embeds          = partial_audio_embeds,
            audio_scale           = config.audio_scale,
            ip_mask               = ip_mask,
            use_un_ip_mask        = config.use_un_ip_mask,
            height                = sample_height,
            width                 = sample_width,
            generator             = generator,
            neg_scale             = config.neg_scale,
            neg_steps             = config.neg_steps,
            use_dynamic_cfg       = config.use_dynamic_cfg,
            use_dynamic_acfg      = config.use_dynamic_acfg,
            guidance_scale        = float(guidance_scale),
            audio_guidance_scale  = float(audio_guidance_scale),
            num_inference_steps   = int(num_inference_steps),
            video                 = input_video,
            mask_video            = input_video_mask,
            clip_image            = clip_image,
            cfg_skip_ratio        = config.cfg_skip_ratio,
            shift                 = config.shift,
        ).videos
        
        if init_frames != 0:
            mix_ratio = torch.from_numpy(
                np.array([float(i) / float(config.overlap_video_length) for i in range(config.overlap_video_length)], np.float32)
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            new_sample[:, :, -config.overlap_video_length:] = (
                new_sample[:, :, -config.overlap_video_length:] * (1 - mix_ratio) +
                sample[:, :, :config.overlap_video_length] * mix_ratio
            )
            new_sample = torch.cat([new_sample, sample[:, :, config.overlap_video_length:]], dim=2)
            sample = new_sample
        else:
            new_sample = sample

        if last_frames >= video_length:
            break

        ref_img = [
            Image.fromarray(
                (sample[0, :, i].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
            ) for i in range(-config.overlap_video_length, 0)
        ]

        init_frames += partial_video_length_adjusted - config.overlap_video_length
        last_frames = init_frames + partial_video_length_adjusted

    progress(0.9, desc="保存视频文件...")
    # Save generated video
    video_path = os.path.join(save_path, f"{timestamp}.mp4")
    video_audio_path = os.path.join(save_path, f"{timestamp}_audio.mp4")

    print ("final sample shape:",sample.shape)
    video_length = sample.shape[2]
    print ("final length:",video_length)

    save_videos_grid(sample[:, :, :video_length], video_path, fps=config.fps)

    progress(0.95, desc="合成音频...")
    video_clip = VideoFileClip(video_path)
    audio_clip = audio_clip.subclipped(0, video_length / config.fps)
    video_clip = video_clip.with_audio(audio_clip)
    video_clip.write_videofile(video_audio_path, codec="libx264", audio_codec="aac", threads=2)

    # 清理临时文件，与官方逻辑保持一致
    try:
        import shutil
        if os.path.exists(video_path):
            os.remove(video_path)  # 使用更安全的删除方式替代 os.system
    except Exception as e:
        print(f"Warning: Failed to clean temporary file {video_path}: {e}")

    progress(1.0, desc="完成！")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return video_audio_path, seed


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
            <div style="text-align: center; padding: 20px;">
                <h1 style="font-size: 36px; color: #2c3e50; margin-bottom: 10px;">🎭 EchoMimicV3</h1>
                <p style="font-size: 18px; color: #7f8c8d;">Audio-Driven Portrait Video Generation with Examples</p>
            </div>
            """)
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 📁 输入设置")
            
            with gr.Group():
                image = gr.Image(
                    label="📷 参考图片", 
                    type="filepath", 
                    height=300,
                    show_label=True
                )
                audio = gr.Audio(
                    label="🎵 音频文件", 
                    type="filepath",
                    show_label=True
                )
            
            with gr.Group():
                prompt = gr.Textbox(
                    label="✨ 提示词", 
                    value="",
                    placeholder="请输入描述视频内容的提示词...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="🚫 负面提示词", 
                    value="Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. 手部快速摆动, 手指频繁抽搐, 夸张手势, 重复机械性动作.",
                    lines=3
                )
                seed_param = gr.Number(
                    label="🎲 随机种子 (-1为随机)", 
                    value=-1,
                    precision=0
                )
            
            # 高级参数控制
            with gr.Accordion("🔧 高级参数设置", open=False):
                gr.Markdown("""
                **参数说明：**
                - **Audio CFG**: 音频引导强度，2-3最佳。增加值提高唇同步，减少值提高视觉质量
                - **Text CFG**: 文本引导强度，3-6最佳。增加值提高提示词跟随，减少值提高视觉质量  
                - **采样步数**: 人头对话5步，全身对话15-25步
                - **TeaCache**: 优化范围0-0.1，可加速推理
                - **视频片段长度**: 较小值减少显存使用，长视频推荐81、65或更小
                """)
                
                with gr.Row():
                    audio_guidance_scale = gr.Slider(
                        minimum=1.0, maximum=5.0, value=2.5, step=0.1,
                        label="🎵 Audio CFG (音频引导强度)",
                        info="推荐范围: 2.0-3.0"
                    )
                    guidance_scale = gr.Slider(
                        minimum=2.0, maximum=8.0, value=4.0, step=0.5,
                        label="📝 Text CFG (文本引导强度)", 
                        info="推荐范围: 3.0-6.0"
                    )
                
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=5, maximum=50, value=20, step=5,
                        label="🔄 采样步数",
                        info="人头对话:5步, 全身对话:15-25步"
                    )
                    partial_video_length = gr.Slider(
                        minimum=25, maximum=97, value=49, step=8,
                        label="📹 视频片段长度",
                        info="较小值减少显存使用"
                    )
                
                with gr.Row():
                    enable_teacache = gr.Checkbox(
                        value=False,
                        label="⚡ 启用 TeaCache 加速",
                        info="可提升推理速度"
                    )
                    teacache_threshold = gr.Slider(
                        minimum=0.0, maximum=0.3, value=0.1, step=0.01,
                        label="🎯 TeaCache 阈值",
                        info="推荐范围: 0.0-0.1"
                    )
            
            gr.Markdown("""
            **💡 提示：**
            - 音频越长，生成时间越久，系统会自动分片段处理
            - 建议音频时长控制在 10-30 秒内以获得最佳体验
            - 生成过程中显示的"片段"是指长视频的分块处理，不是多个视频
            """)
            
            generate_button = gr.Button(
                "🎬 开始生成视频", 
                variant='primary',
                size='lg'
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 📺 生成结果")
            
            with gr.Group():
                video_output = gr.Video(
                    label="🎥 生成的视频", 
                    interactive=False,
                    height=400
                )
                seed_output = gr.Textbox(
                    label="🔢 使用的种子值",
                    interactive=False
                )
    
    # 示例区域放在底部，占据全宽
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎭 演示样例")
            gr.Markdown("点击下方任意样例快速填充输入字段，然后点击生成按钮开始处理")
            
            if DEMO_EXAMPLES:
                gr.Markdown(f"**📊 共加载 {len(DEMO_EXAMPLES)} 个演示样例**")
                example_component = gr.Examples(
                    examples=DEMO_EXAMPLES,
                    inputs=[image, audio, prompt, negative_prompt, seed_param],
                    label=None,
                    examples_per_page=10  # 每页显示10个示例
                )
            else:
                gr.Markdown("*❌ 暂无可用的演示样例*")

    gr.on(
        triggers=[generate_button.click],
        fn=generate,
        inputs=[
            image,
            audio,
            prompt,
            negative_prompt,
            seed_param,
            audio_guidance_scale,
            guidance_scale,
            num_inference_steps,
            teacache_threshold,
            partial_video_length,
            enable_teacache,
        ],
        outputs=[video_output, seed_output]
    )
        

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )
