import os
import gradio as gr
import torch
import numpy as np
import glob
from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange
from pipeline import CausalInferencePipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import *
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file
import tempfile
import shutil

class InteractiveGameInference:
    def __init__(self, config_path, checkpoint_path, pretrained_model_path):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.pretrained_model_path = pretrained_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load(self.config_path)

    def _init_models(self):
        # Initialize pipeline
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(self.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        pipeline = CausalInferencePipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)
        if self.checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(self.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(self.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image
    
    def generate_videos(self, img_path, num_output_frames, seed, output_folder):
        set_seed(seed)
        os.makedirs(output_folder, exist_ok=True)
        
        mode = self.config.get('mode', 'universal')
        
        image = load_image(img_path)
        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)

        # Encode the input image as the first latent
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1) 
        visual_context = self.vae.clip.encode_video(image)
        sampled_noise = torch.randn(
            [1, 16, num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype
        )
        num_frames = (num_output_frames - 1) * 4 + 1
        
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }
        
        if mode == 'universal':
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        elif mode == 'gta_drive':
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict['keyboard_cond'] = keyboard_condition
        
        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                mode=mode,
                profile=False
            )

        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)
        mouse_icon = 'assets/images/mouse.png'
        
        # 生成输出视频文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_folder, f'{base_name}_demo.mp4')
        output_icon_path = os.path.join(output_folder, f'{base_name}_demo_icon.mp4')
        
        if mode != 'templerun':
            config = (
                keyboard_condition[0].float().cpu().numpy(),
                mouse_condition[0].float().cpu().numpy()
            )
        else:
            config = (
                keyboard_condition[0].float().cpu().numpy()
            )
        
        process_video(video.astype(np.uint8), output_path, config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
        process_video(video.astype(np.uint8), output_icon_path, config, mouse_icon, mouse_scale=0.1, process_icon=True, mode=mode)
        
        return output_path, output_icon_path

def get_demo_images():
    """获取所有demo图片"""
    demo_images = []
    for folder in ['universal', 'gta_drive', 'temple_run']:
        folder_path = f'demo_images/{folder}'
        if os.path.exists(folder_path):
            images = glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.webp'))
            demo_images.extend(sorted(images))
    return demo_images

def get_config_files():
    """获取所有配置文件"""
    config_files = glob.glob('configs/inference_yaml/*.yaml')
    return sorted(config_files)

def get_checkpoint_paths():
    """获取所有checkpoint路径"""
    checkpoints = []
    base_path = "Matrix-Game-2.0"
    if os.path.exists(base_path):
        for subfolder in ['gta_distilled_model', 'templerun_distilled_model', 'base_distilled_model']:
            subfolder_path = os.path.join(base_path, subfolder)
            if os.path.exists(subfolder_path):
                safetensors_files = glob.glob(os.path.join(subfolder_path, '*.safetensors'))
                checkpoints.extend(safetensors_files)
    return sorted(checkpoints)

def inference_fn(image, config_path, checkpoint_path, num_output_frames, seed, pretrained_model_path):
    """推理函数"""
    if not image:
        return None, None, "❌ 请先上传图片"
    
    if not config_path:
        return None, None, "❌ 请选择配置文件"
    
    if not checkpoint_path:
        return None, None, "❌ 请选择模型权重"
    
    if not os.path.exists(config_path):
        return None, None, f"❌ 配置文件不存在: {config_path}"
    
    if not os.path.exists(checkpoint_path):
        return None, None, f"❌ 模型权重文件不存在: {checkpoint_path}"
    
    try:
        # 创建临时输出文件夹
        with tempfile.TemporaryDirectory() as temp_dir:
            # 如果image是上传的文件，需要处理路径
            if hasattr(image, 'name'):
                img_path = image.name
            else:
                img_path = image
            
            if not os.path.exists(img_path):
                return None, None, f"❌ 图片文件不存在: {img_path}"
            
            print(f"🔄 开始推理...")
            print(f"📷 输入图片: {img_path}")
            print(f"⚙️ 配置文件: {config_path}")
            print(f"🔧 模型权重: {checkpoint_path}")
            
            # 初始化推理器
            inferencer = InteractiveGameInference(config_path, checkpoint_path, pretrained_model_path)
            
            # 生成视频
            output_path, output_icon_path = inferencer.generate_videos(
                img_path=img_path,
                num_output_frames=num_output_frames,
                seed=seed,
                output_folder=temp_dir
            )
            
            # 复制到输出文件夹
            os.makedirs("outputs", exist_ok=True)
            final_output = os.path.join("outputs", os.path.basename(output_path))
            final_icon_output = os.path.join("outputs", os.path.basename(output_icon_path))
            shutil.copy2(output_path, final_output)
            shutil.copy2(output_icon_path, final_icon_output)
            
            print(f"✅ 推理完成!")
            print(f"📁 输出文件: {final_output}")
            print(f"📁 带图标版本: {final_icon_output}")
            
            return final_output, final_icon_output, "✅ 推理完成！视频已保存到 outputs/ 文件夹"
            
    except Exception as e:
        error_msg = f"❌ 推理失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, error_msg

def create_examples():
    """创建示例数据"""
    demo_images = get_demo_images()
    config_files = get_config_files()
    checkpoint_paths = get_checkpoint_paths()
    
    examples = []
    
    # 为每种模式创建示例
    modes = {
        'universal': ('configs/inference_yaml/inference_universal.yaml', 'Matrix-Game-2.0/base_distilled_model'),
        'gta_drive': ('configs/inference_yaml/inference_gta_drive.yaml', 'Matrix-Game-2.0/gta_distilled_model'),
        'templerun': ('configs/inference_yaml/inference_templerun.yaml', 'Matrix-Game-2.0/templerun_distilled_model')
    }
    
    for mode, (config, checkpoint_folder) in modes.items():
        # 找到对应模式的图片
        mode_images = [img for img in demo_images if mode.replace('_', '') in img or mode in img]
        if not mode_images:
            continue
            
        # 找到对应的checkpoint
        checkpoint_files = glob.glob(os.path.join(checkpoint_folder, '*.safetensors'))
        if not checkpoint_files:
            continue
            
        # 创建示例
        example_image = mode_images[0] if mode_images else None
        example_checkpoint = checkpoint_files[0] if checkpoint_files else ""
        
        if example_image and os.path.exists(config):
            examples.append([
                example_image,
                config,
                example_checkpoint,
                150,  # num_output_frames
                42,   # seed
                "Matrix-Game-2.0"  # pretrained_model_path
            ])
    
    return examples

def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="Matrix Game Inference WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎮 Matrix Game Inference WebUI
        
        将单张游戏截图转换为连续游戏视频的AI推理工具
        
        支持三种游戏模式：
        - **Universal（通用）**: 适用于各种游戏场景
        - **GTA Drive（GTA驾驶）**: 专门针对GTA类驾驶游戏优化  
        - **Temple Run（神庙逃亡）**: 专门针对跑酷类游戏优化
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 输入设置")
                
                image_input = gr.Image(
                    label="输入图片 (支持 PNG、JPG、WEBP 格式)",
                    type="filepath",
                    height=300
                )
                
                config_dropdown = gr.Dropdown(
                    choices=get_config_files(),
                    label="配置文件 (选择对应的游戏模式)",
                    value="configs/inference_yaml/inference_universal.yaml" if os.path.exists("configs/inference_yaml/inference_universal.yaml") else None
                )
                
                checkpoint_dropdown = gr.Dropdown(
                    choices=get_checkpoint_paths(),
                    label="模型权重 (选择对应模式的权重文件)",
                    value=get_checkpoint_paths()[0] if get_checkpoint_paths() else ""
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 参数设置")
                
                num_frames_slider = gr.Slider(
                    minimum=10,
                    maximum=300,
                    value=150,
                    step=1,
                    label="输出帧数 (更多帧数=更长视频，但需要更多时间和内存)"
                )
                
                seed_number = gr.Number(
                    value=42,
                    label="随机种子 (相同种子产生相同结果)",
                    precision=0
                )
                
                pretrained_path = gr.Textbox(
                    value="Matrix-Game-2.0",
                    label="预训练模型路径",
                    info="包含 VAE 和其他基础模型的文件夹"
                )
                
                run_button = gr.Button("🚀 开始推理", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📺 输出结果")
                with gr.Row():
                    output_video = gr.Video(label="生成视频（普通版）", height=300)
                    output_icon_video = gr.Video(label="生成视频（带操作图标版）", height=300)
                status_output = gr.Textbox(label="状态信息", interactive=False, lines=3)
        
        # 示例
        gr.Markdown("""
        ### 💡 快速开始示例
        点击下面的示例可以快速加载对应的配置和图片
        """)
        examples = create_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[
                    image_input,
                    config_dropdown,
                    checkpoint_dropdown,
                    num_frames_slider,
                    seed_number,
                    pretrained_path
                ],
                label="点击示例快速开始"
            )
        else:
            gr.Markdown("⚠️ 未找到示例文件，请检查 demo_images 和模型文件是否存在")
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 使用步骤：
            1. **选择图片**: 上传一张游戏截图，建议分辨率不要太高
            2. **选择配置**: 根据游戏类型选择对应的 yaml 配置文件
            3. **选择权重**: 选择与配置文件匹配的模型权重文件
            4. **调整参数**: 设置输出帧数和随机种子
            5. **开始推理**: 点击按钮开始生成视频
            
            ### 注意事项：
            - 推理过程需要一定时间，请耐心等待
            - 更多帧数会产生更长的视频，但需要更多GPU内存
            - 如果遇到内存不足，请减少输出帧数
            - 生成的视频会保存在 `outputs/` 文件夹中
            """)
        
        # 绑定推理函数
        run_button.click(
            fn=inference_fn,
            inputs=[
                image_input,
                config_dropdown,
                checkpoint_dropdown,
                num_frames_slider,
                seed_number,
                pretrained_path
            ],
            outputs=[output_video, output_icon_video, status_output],
            show_progress=True
        )
    
    return demo

def check_environment():
    """检查运行环境"""
    print("🎮 Matrix Game WebUI")
    print("=" * 50)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA 不可用，将使用CPU运行（速度较慢）")
    
    # 检查必要文件
    required_files = [
        "Matrix-Game-2.0/Wan2.1_VAE.pth",
        "assets/images/mouse.png"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n请确保已正确下载所有模型文件")
        return False
    
    # 检查配置文件
    config_files = get_config_files()
    if not config_files:
        print("❌ 未找到配置文件，请检查 configs/inference_yaml/ 目录")
        return False
    
    # 检查checkpoint文件
    checkpoint_files = get_checkpoint_paths()
    if not checkpoint_files:
        print("❌ 未找到模型权重文件，请检查 Matrix-Game-2.0/ 目录")
        return False
    
    print("✅ 环境检查通过")
    print(f"📁 找到 {len(config_files)} 个配置文件")
    print(f"🔧 找到 {len(checkpoint_files)} 个模型权重")
    print(f"🖼️  找到 {len(get_demo_images())} 个示例图片")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    # 检查环境
    if not check_environment():
        print("环境检查失败，程序退出")
        exit(1)
    
    # 确保输出文件夹存在
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # 创建并启动界面
        print("🚀 启动WebUI...")
        print("界面将在 http://localhost:7860 打开")
        print("按 Ctrl+C 停止服务")
        print("=" * 50)
        
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 WebUI 已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {str(e)}")
        print("请检查依赖是否正确安装")
