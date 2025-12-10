# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Single GPU version for LiveAvatar Gradio App
# Adapted for single H20 GPU with 141GB VRAM

import logging
import os
import sys
import warnings
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

import random
import gradio as gr
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image

from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from liveavatar.models.wan.wan_2_2.utils.utils import merge_video_audio, save_video
from liveavatar.utils.args_config import parse_args_for_training_config

# Global variables for pipeline and config
wan_s2v_pipeline = None
global_cfg = None
global_training_settings = None

# Default configuration
DEFAULT_CONFIG = {
    "task": "s2v-14B",
    "size": "704*384",  # Smaller size for single GPU
    "base_seed": 420,
    "infer_frames": 48,
    "sample_steps": 4,  # Fixed at 4 - pipeline KV cache is designed for 4 steps
    "sample_guide_scale": 4.5,  # Official default, higher = more stable
    "num_clip": 10000,  # Will be auto-adjusted based on audio length
    "sample_solver": "euler",
    "offload_model": False,  # Disabled - 141GB VRAM is enough
    "ckpt_dir": "checkpoints/Wan2.2-S2V-14B/",
    "lora_path": "checkpoints/Live-Avatar/liveavatar.safetensors",
    "training_config": "liveavatar/configs/s2v_causal_sft.yaml",
    "server_port": 7860,
    "server_name": "0.0.0.0",
    "save_dir": "./output/gradio/",
}

EXAMPLE_PROMPT = {
    "s2v-14B": {
        "prompt":
            "A professional female news anchor speaking to the camera with composed posture. Steady camera, static shot, minimal hand movement, hands naturally at rest. Clean studio background, professional lighting, smooth subtle facial expressions. High quality, realistic.",
        "image":
            "examples/anchor.jpg",
        "audio":
            "examples/fashion_blogger.wav",
    },
}

# Example images for gallery
EXAMPLE_IMAGES = [
    ("examples/anchor.jpg", "Anchor"),
    ("examples/fashion_blogger.jpg", "Fashion Blogger"),
    ("examples/kitchen_grandmother.jpg", "Kitchen Grandmother"),
    ("examples/music_producer.jpg", "Music Producer"),
    ("examples/dwarven_blacksmith.jpg", "Dwarven Blacksmith"),
    ("examples/cyclops_baker.jpg", "Cyclops Baker"),
    ("examples/livestream_1.png", "Livestream 1"),
    ("examples/livestream_2.jpg", "Livestream 2"),
]

# Example audios
EXAMPLE_AUDIOS = [
    ("examples/fashion_blogger.wav", "Fashion Blogger Audio"),
    ("examples/kitchen_grandmother.wav", "Kitchen Grandmother Audio"),
    ("examples/music_producer.wav", "Music Producer Audio"),
    ("examples/dwarven_blacksmith.wav", "Dwarven Blacksmith Audio"),
    ("examples/cyclops_baker.wav", "Cyclops Baker Audio"),
]


def _init_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])


def initialize_pipeline():
    """
    Initialize the LiveAvatar pipeline for single GPU
    """
    global wan_s2v_pipeline, global_cfg, global_training_settings
    
    device = torch.device("cuda:0")
    
    # Initialize a fake single-process distributed environment
    # This is needed because the pipeline code uses dist.get_rank() in some places
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        logging.info("Initialized single-process distributed environment")
    
    # Load training settings from config
    training_settings = parse_args_for_training_config(DEFAULT_CONFIG["training_config"])
    global_training_settings = training_settings
    
    cfg = WAN_CONFIGS[DEFAULT_CONFIG["task"]]
    global_cfg = cfg
    
    logging.info(f"Initializing single GPU pipeline on {device}")
    logging.info(f"Model config: {cfg}")
    logging.info(f"Checkpoint dir: {DEFAULT_CONFIG['ckpt_dir']}")
    logging.info(f"LoRA path: {DEFAULT_CONFIG['lora_path']}")
    
    # Import single GPU pipeline
    from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
    
    logging.info("Creating WanS2V pipeline...")
    wan_s2v = WanS2V(
        config=cfg,
        checkpoint_dir=DEFAULT_CONFIG["ckpt_dir"],
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        sp_size=1,
        t5_cpu=False,  # With 141GB VRAM, we can keep T5 on GPU
        init_on_cpu=False,  # Keep models on GPU, don't init on CPU
        convert_model_dtype=True,
        single_gpu=True,
        offload_kv_cache=False,  # With enough VRAM, no need to offload
    )
    
    # Load LoRA weights
    lora_path = DEFAULT_CONFIG["lora_path"]
    if os.path.exists(lora_path):
        logging.info(f'Loading LoRA: path={lora_path}, rank={training_settings["lora_rank"]}, alpha={training_settings["lora_alpha"]}')
        wan_s2v.add_lora_to_model(
            wan_s2v.noise_model,
            lora_rank=training_settings['lora_rank'],
            lora_alpha=training_settings['lora_alpha'],
            lora_target_modules=training_settings['lora_target_modules'],
            init_lora_weights=training_settings['init_lora_weights'],
            pretrained_lora_path=lora_path,
            load_lora_weight_only=False,
        )
    else:
        logging.warning(f"LoRA path not found: {lora_path}")
    
    # When offload_model=False, ensure VAE is on GPU
    # (pipeline code only moves VAE to GPU when offload_model=True)
    if not DEFAULT_CONFIG["offload_model"]:
        wan_s2v.vae.model.to(wan_s2v.device)
        logging.info("Moved VAE to GPU (offload_model=False)")
    
    wan_s2v_pipeline = wan_s2v
    logging.info("Pipeline initialized successfully!")


def run_inference(prompt, image_path, audio_path, num_clip, 
                  sample_steps, sample_guide_scale, infer_frames,
                  size, base_seed, sample_solver):
    """
    Run inference for a single sample on single GPU
    """
    global wan_s2v_pipeline, global_cfg
    
    try:
        logging.info(f"Generating video...")
        logging.info(f"  Prompt: {prompt}")
        logging.info(f"  Image: {image_path}")
        logging.info(f"  Audio: {audio_path}")
        logging.info(f"  Num_clip: {num_clip}")
        logging.info(f"  Sample_steps: {sample_steps}")
        logging.info(f"  Guide_scale: {sample_guide_scale}")
        logging.info(f"  Infer_frames: {infer_frames}")
        logging.info(f"  Size: {size}")
        logging.info(f"  Seed: {base_seed}")
        logging.info(f"  Solver: {sample_solver}")
        
        # Generate video
        video, dataset_info = wan_s2v_pipeline.generate(
            input_prompt=prompt,
            ref_image_path=image_path,
            audio_path=audio_path,
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=num_clip,
            pose_video=None,
            generate_size=size,
            max_area=MAX_AREA_CONFIGS.get(size, 704 * 384),
            infer_frames=infer_frames,
            shift=3.0,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=sample_guide_scale,
            seed=base_seed,
            offload_model=DEFAULT_CONFIG["offload_model"],
            init_first_frame=False,
            use_dataset=False,
            dataset_sample_idx=0,
            drop_motion_noisy=False,
            num_gpus_dit=1,
            enable_vae_parallel=False,
            input_video_for_sam2=None,
        )
        
        logging.info("Denoising video done")
        
        # Save video
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        suffix = '.mp4'
        
        save_file = f"{formatted_time}_{sample_steps}step_{formatted_prompt}"
        
        save_dir = DEFAULT_CONFIG["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, save_file + suffix)
        
        logging.info(f"Saving generated video to {save_file}")
        save_video(
            tensor=video[None],
            save_file=save_file,
            fps=global_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        
        # Merge audio
        merge_video_audio(video_path=save_file, audio_path=audio_path)
        
        video_path = save_file
        logging.info(f"Video saved successfully: {video_path}")
        
        # Clean up
        del video
        torch.cuda.empty_cache()
        
        return video_path
        
    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        logging.error(error_msg)
        import traceback
        traceback.print_exc()
        return None


def create_gradio_interface():
    """
    Create Gradio web interface
    """
    with gr.Blocks(title="LiveAvatar Video Generation") as demo:
        gr.Markdown("# LiveAvatar 视频生成 Web UI / LiveAvatar Video Generation Web UI")
        gr.Markdown("上传参考图像、音频和提示词来生成说话人视频 / Upload reference image, audio and prompt to generate talking avatar video")
        gr.Markdown("**单 GPU 模式 (H20 141GB) / Single GPU Mode (H20 141GB)**")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 基础输入 / Basic Input")
                prompt_input = gr.Textbox(
                    label="提示词 / Prompt",
                    placeholder="描述你想生成的视频内容 / Describe the video content you want to generate...",
                    value=EXAMPLE_PROMPT["s2v-14B"]["prompt"],
                    lines=5
                )
                
                # Image input with gallery
                image_input = gr.Image(
                    label="参考图像 / Reference Image",
                    type="filepath"
                )
                
                gr.Markdown("**示例图片 (点击选择) / Example Images (Click to Select):**")
                example_gallery = gr.Gallery(
                    value=[img_path for img_path, label in EXAMPLE_IMAGES if os.path.exists(img_path)],
                    label="",
                    show_label=False,
                    columns=4,
                    rows=2,
                    height=200,
                    object_fit="cover"
                )
                
                # Audio input with examples
                audio_input = gr.Audio(
                    label="音频文件 / Audio File",
                    type="filepath"
                )
                
                example_audio_dropdown = gr.Dropdown(
                    choices=[(label, audio_path) for audio_path, label in EXAMPLE_AUDIOS if os.path.exists(audio_path)],
                    label="示例音频 (选择后自动填充) / Example Audio (Auto-fill on Selection)",
                    show_label=True,
                    value=None
                )
                
                with gr.Accordion("高级参数 / Advanced Parameters", open=False):
                    gr.Markdown("### 生成参数 / Generation Parameters")
                    with gr.Row():
                        num_clip_input = gr.Slider(
                            minimum=1,
                            maximum=10000,
                            value=DEFAULT_CONFIG["num_clip"],
                            step=1,
                            label="生成片段数量 / Number of Clips"
                        )
                        sample_steps_input = gr.Slider(
                            minimum=4,
                            maximum=4,
                            value=4,
                            step=1,
                            label="采样步数 / Sampling Steps (固定为4 / Fixed at 4)",
                            interactive=False  # Disable - only 4 steps supported
                        )
                    
                    with gr.Row():
                        sample_guide_scale_input = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            value=DEFAULT_CONFIG["sample_guide_scale"],
                            step=0.1,
                            label="引导尺度 / Guidance Scale"
                        )
                        infer_frames_input = gr.Slider(
                            minimum=16,
                            maximum=160,
                            value=DEFAULT_CONFIG["infer_frames"],
                            step=4,
                            label="每片段帧数 / Frames per Clip"
                        )
                    
                    with gr.Row():
                        size_input = gr.Dropdown(
                            choices=list(SIZE_CONFIGS.keys()),
                            value=DEFAULT_CONFIG["size"],
                            label="视频尺寸 / Video Size"
                        )
                        base_seed_input = gr.Number(
                            value=DEFAULT_CONFIG["base_seed"],
                            label="随机种子 / Random Seed",
                            precision=0
                        )
                
                generate_btn = gr.Button("🎬 开始生成 / Start Generation", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### 生成结果 / Generation Result")
                video_output = gr.Video(label="生成的视频 / Generated Video")
                status_output = gr.Textbox(label="状态信息 / Status", lines=3)
        
        # Add example combinations
        gr.Markdown("### 📌 快速示例 / Quick Examples")
        gr.Examples(
            examples=[
                [
                    "A young boy with curly brown hair and a white polo shirt, wearing a backpack, standing outdoors in a sunny park. He is speaking animatedly with expressive hand gestures, his hands clearly visible and moving naturally. His facial expressions are lively and engaging, conveying enthusiasm and curiosity. The camera captures him from the chest up with a steady, clear shot, natural lighting and a soft green background. Realistic 3D animation style, lifelike motion and expression.",
                    "examples/boy.jpg",
                    "examples/boy.wav",
                    10000,
                    4,
                    0.0,
                    48,
                    "704*384",
                    420,
                ],
                [
                    "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge.",
                    "examples/dwarven_blacksmith.jpg",
                    "examples/dwarven_blacksmith.wav",
                    10,
                    4,
                    0.0,
                    48,
                    "704*384",
                    420,
                ],
            ],
            inputs=[
                prompt_input, image_input, audio_input, num_clip_input,
                sample_steps_input, sample_guide_scale_input, infer_frames_input,
                size_input, base_seed_input
            ],
            examples_per_page=2
        )
        
        def generate_wrapper(prompt, image, audio, num_clip, sample_steps, 
                           sample_guide_scale, infer_frames, size, base_seed):
            """Wrapper function for Gradio interface"""
            if not prompt or not image or not audio:
                return None, "错误 / Error: 请提供所有必需的输入 (提示词、图像、音频) / Please provide all required inputs (prompt, image, audio)"
            
            try:
                status = f"正在生成视频 / Generating video...\n参数 / Parameters: steps={sample_steps}, clips={num_clip}, frames={infer_frames}"
                video_path = run_inference(
                    prompt, image, audio, num_clip,
                    sample_steps, sample_guide_scale, infer_frames,
                    size, int(base_seed), "euler"  # 固定使用 euler 采样器
                )
                
                if video_path and os.path.exists(video_path):
                    status = f"✅ 生成成功 / Generation Successful!\n视频保存在 / Video saved at: {video_path}"
                    return video_path, status
                else:
                    status = "❌ 生成失败，请查看日志 / Generation failed, please check logs"
                    return None, status
            except Exception as e:
                status = f"❌ 错误 / Error: {str(e)}"
                return None, status
        
        def select_example_image(evt: gr.SelectData):
            """Handle example image selection"""
            selected_index = evt.index
            if selected_index < len(EXAMPLE_IMAGES):
                img_path = EXAMPLE_IMAGES[selected_index][0]
                if os.path.exists(img_path):
                    return img_path
            return None
        
        def select_example_audio(audio_path):
            """Handle example audio selection"""
            if audio_path and os.path.exists(audio_path):
                return audio_path
            return None
        
        # Connect event handlers
        example_gallery.select(
            fn=select_example_image,
            outputs=image_input
        )
        
        example_audio_dropdown.change(
            fn=select_example_audio,
            inputs=example_audio_dropdown,
            outputs=audio_input
        )
        
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                prompt_input, image_input, audio_input, num_clip_input,
                sample_steps_input, sample_guide_scale_input, infer_frames_input,
                size_input, base_seed_input
            ],
            outputs=[video_output, status_output]
        )
    
    return demo


def main():
    """Main entry point"""
    # Initialize logging first
    _init_logging()
    
    logging.info("=" * 50)
    logging.info("Starting LiveAvatar Single GPU Gradio App")
    logging.info("=" * 50)
    
    # Initialize pipeline
    logging.info("Initializing pipeline...")
    initialize_pipeline()
    
    # Launch Gradio interface
    logging.info(f"Launching LiveAvatar Gradio interface on {DEFAULT_CONFIG['server_name']}:{DEFAULT_CONFIG['server_port']}")
    demo = create_gradio_interface()
    demo.launch(
        server_name=DEFAULT_CONFIG["server_name"],
        server_port=DEFAULT_CONFIG["server_port"],
        share=False
    )


if __name__ == "__main__":
    main()
