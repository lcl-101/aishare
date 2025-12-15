"""
SCAIL Gradio Web Application
A web interface for character animation using SCAIL model.
"""

import os
import sys
import math
import argparse
import tempfile
import shutil
from typing import List, Union
from functools import wraps
import numpy as np
import torch

# Fix for PyTorch 2.6+ weights_only default change
# Monkey patch torch.load to use weights_only=False by default
_original_torch_load = torch.load

@wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from einops import rearrange
from PIL import Image
import imageio
import gradio as gr
import decord
from decord import VideoReader
import torchvision.transforms as TT
import torch.nn.functional as F

# Set environment variables for distributed training (single GPU mode)
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["LOCAL_WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

from omegaconf import OmegaConf, ListConfig
from sgm.util import get_obj_from_str, exists
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
import diffusion_video


# ======================= Helper Functions =======================

def resize_for_rectangle_crop(arr, image_size, reshape_mode="center"):
    """Resize and crop array to target size"""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize
    
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        top, left = delta_h // 2, delta_w // 2
        
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def load_image_to_tensor_chw_normalized(image_data):
    """Load image and convert to normalized tensor"""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert('RGB')
    else:
        image = Image.fromarray(image_data).convert('RGB')
    transform = TT.Compose([TT.ToTensor()])
    image_tensor = transform(image)
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0)  # 1 C H W, -1 to 1
    return image_tensor


def load_video_for_pose_sample(video_data):
    """Load video and convert to tensor"""
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    indices = np.arange(0, len(vr))
    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    return tensor_frms


def get_unique_embedder_keys_from_conditioner(conditioner):
    """Get unique keys from conditioner embedders"""
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    """Prepare batch for model inference"""
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = rearrange(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = torch.tensor([value_dict["cond_aug"]]).to("cuda").repeat(math.prod(N))
        elif key == "cond_frames":
            batch[key] = value_dict["cond_frames"].repeat(N[0], 1, 1, 1, 1)
        elif key == "cond_frames_without_noise":
            batch[key] = value_dict["cond_frames_without_noise"].repeat(N[0], 1, 1, 1, 1)
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


# ======================= Model Manager =======================

class SCAILModel:
    """Manages SCAIL model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.args = None
        self.loaded = False
        
    def load_model(self, checkpoint_path="checkpoints/SCAIL-Preview"):
        """Load the SCAIL model"""
        if self.loaded:
            return
        
        print("Loading SCAIL model...")
        
        # Get the absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(base_dir, checkpoint_path)
        
        # Build config paths
        model_config_path = os.path.join(base_dir, "configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml")
        sampling_config_path = os.path.join(base_dir, "configs/sampling/wan_pose_14Bsc_xc_cli.yaml")
        
        # Update config paths to use checkpoint_path
        model_config = OmegaConf.load(model_config_path)
        sampling_config = OmegaConf.load(sampling_config_path)
        
        # Update paths in model config
        model_config.model.conditioner_config.params.emb_models[0].params.checkpoint_path = \
            os.path.join(checkpoint_path, "umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth")
        model_config.model.conditioner_config.params.emb_models[0].params.tokenizer_path = \
            os.path.join(checkpoint_path, "umt5-xxl")
        model_config.model.i2v_clip_config.params.checkpoint_path = \
            os.path.join(checkpoint_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth")
        model_config.model.first_stage_config.params.vae_pth = \
            os.path.join(checkpoint_path, "Wan2.1_VAE.pth")
        
        # Update load path in sampling config
        sampling_config.args.load = os.path.join(checkpoint_path, "model")
        
        # Merge configs
        config = OmegaConf.merge(model_config, sampling_config)
        
        # Build args manually
        args_dict = OmegaConf.to_object(config.get("args", {}))
        
        # Use argparse.Namespace for compatibility with SAT library
        base_args = argparse.Namespace()
        
        # Set default values
        base_args.mode = "inference"
        base_args.seed = 42
        base_args.batch_size = 1
        base_args.train_data = None
        base_args.train_iters = None
        base_args.epochs = None
        base_args.lr = 1e-4
        base_args.weight_decay = 0.0
        base_args.gradient_accumulation_steps = 1
        base_args.local_rank = 0
        base_args.fp16 = False
        base_args.bf16 = True
        base_args.deepspeed_config = None
        base_args.zero_stage = 0
        base_args.checkpoint_activations = False
        base_args.model_parallel_size = 1
        base_args.sequence_parallel_size = 1
        base_args.num_multi_query_heads = 0
        base_args.distributed_backend = "nccl"
        base_args.train_data_weights = None
        base_args.force_inference = True  # Important for inference mode
        base_args.rank = 0
        base_args.world_size = 1
        base_args.cuda = torch.cuda.is_available()
        base_args.device = 0 if torch.cuda.is_available() else "cpu"
        base_args.deepspeed = False
        
        # Override with config args
        for key, value in args_dict.items():
            setattr(base_args, key, value)
        
        # Set model config
        base_args.model_config = config.get("model", {})
        
        # Disable checkpoint activations for inference
        if hasattr(base_args, 'model_config') and 'network_config' in base_args.model_config:
            base_args.model_config.network_config.params.transformer_args.checkpoint_activations = False
        
        self.args = base_args
        
        # Initialize distributed
        self._initialize_distributed()
        
        # Create model
        Engine = diffusion_video.SATVideoDiffusionEngine
        self.model = get_model(self.args, Engine)
        
        # Load checkpoint
        if self.args.load is not None:
            load_checkpoint(self.model, self.args)
        
        self.model.eval()
        self.loaded = True
        print("Model loaded successfully!")
    
    def _initialize_distributed(self):
        """Initialize distributed training environment for single GPU"""
        if torch.distributed.is_initialized():
            return
        
        # Set device
        if self.args.device != "cpu":
            torch.cuda.set_device(f"cuda:{self.args.device}")
        
        # Initialize process group
        torch.distributed.init_process_group(
            backend=self.args.distributed_backend,
            world_size=1,
            rank=0,
            init_method="tcp://localhost:29500",
        )
        
        # Initialize model parallel
        mpu.initialize_model_parallel(
            self.args.model_parallel_size,
            self.args.sequence_parallel_size,
            self.args.num_multi_query_heads,
        )
        
        # Initialize context parallel
        from sgm.util import initialize_context_parallel
        initialize_context_parallel(1)
        
    def generate(self, ref_image, pose_video, prompt, negative_prompt="", 
                 sampling_fps=16, num_steps=50, cfg_scale=4.0, progress=gr.Progress()):
        """Generate animated video"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        progress(0, desc="Preparing inputs...")
        
        # Get target size from args
        target_H, target_W = self.args.sampling_image_size
        
        # Load and process reference image
        if isinstance(ref_image, np.ndarray):
            image_tensor = load_image_to_tensor_chw_normalized(ref_image)
        else:
            image_tensor = load_image_to_tensor_chw_normalized(ref_image)
        
        # Adjust orientation
        if image_tensor.shape[2] < image_tensor.shape[3]:
            target_H, target_W = self.args.sampling_image_size
        else:
            target_W, target_H = self.args.sampling_image_size
        
        progress(0.1, desc="Loading pose video...")
        
        # Load pose video
        decord.bridge.set_bridge("torch")
        vr_for_fps = VideoReader(uri=pose_video, height=-1, width=-1)
        driving_fps = vr_for_fps.get_avg_fps()
        
        pose_video_tensor = load_video_for_pose_sample(pose_video)
        pose_video_tensor = pose_video_tensor.permute(0, 3, 1, 2)  # T H W C -> T C H W
        pose_video_tensor = resize_for_rectangle_crop(pose_video_tensor, [target_H, target_W], reshape_mode="center")
        pose_video_tensor = (pose_video_tensor - 127.5) / 127.5  # 0-255 -> -1 to 1
        
        # Process reference image
        image_tensor = resize_for_rectangle_crop(image_tensor, [target_H, target_W], reshape_mode="center")
        
        progress(0.2, desc="Encoding inputs...")
        
        with torch.no_grad():
            # Prepare for VAE encoding
            pose_video_tensor = pose_video_tensor.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B T C H W
            
            if "smpl_downsample" in self.args.representation:
                smpl_render_video = F.interpolate(
                    pose_video_tensor.squeeze(0), 
                    scale_factor=0.5, 
                    mode='bilinear', 
                    align_corners=False
                ).unsqueeze(0)  # 1 T C H W
            else:
                smpl_render_video = pose_video_tensor
            
            ori_image = image_tensor.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B 1 C H W
            
            # Create image sequence (first frame + zeros)
            image = torch.concat([ori_image, torch.zeros_like(pose_video_tensor[:, 1:])], dim=1)
            image = rearrange(image, 'b t c h w -> b c t h w').contiguous()
            image = self.model.encode_first_stage(image, None, force_encode=True)
            image = image.permute(0, 2, 1, 3, 4).contiguous()  # BCTHW -> BTCHW
            
            # Encode reference
            ref_concat = self.model.encode_first_stage(
                rearrange(ori_image, 'b t c h w -> b c t h w').contiguous(), 
                None, 
                force_encode=True
            )
            ref_concat = ref_concat.permute(0, 2, 1, 3, 4).contiguous()
            
            # Encode pose
            if "smpl" in self.args.representation:
                smpl_render_latent = self.model.encode_first_stage(
                    rearrange(smpl_render_video, 'b t c h w -> b c t h w').contiguous(), 
                    None, 
                    force_encode=True
                )
                smpl_render_latent = smpl_render_latent.permute(0, 2, 1, 3, 4).contiguous()
                pose_latent = smpl_render_latent
            else:
                pose_latent = self.model.encode_first_stage(
                    rearrange(pose_video_tensor, 'b t c h w -> b c t h w').contiguous(), 
                    None, 
                    force_encode=True
                )
                pose_latent = pose_latent.permute(0, 2, 1, 3, 4).contiguous()
            
            T = pose_latent.shape[1]
            C, H, W = image.shape[2], image.shape[3], image.shape[4]
            
            # Get CLIP features
            if self.model.use_i2v_clip:
                self.model.i2v_clip.model.to('cuda')
                image_clip_features = self.model.i2v_clip.visual(ori_image.permute(0, 2, 1, 3, 4))
                self.model.i2v_clip.model.cpu()
            
            progress(0.3, desc="Preparing conditions...")
            
            # Prepare value dict
            value_dict = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'num_frames': torch.tensor(T).unsqueeze(0)
            }
            
            num_samples = [1]
            force_uc_zero_embeddings = []
            
            # Get text embeddings
            self.model.conditioner.embedders[0].to('cuda')
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(self.model.conditioner),
                value_dict,
                num_samples
            )
            
            c, uc = self.model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            self.model.conditioner.embedders[0].cpu()
            
            # Move conditions to cuda
            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(
                        lambda y: y[k][:math.prod(num_samples)].to("cuda"), (c, uc)
                    )
            
            # Add image conditions
            c["concat_images"] = image
            uc["concat_images"] = image
            c["ref_concat"] = ref_concat
            uc["ref_concat"] = ref_concat
            c["concat_pose"] = pose_latent
            uc["concat_pose"] = pose_latent
            
            if "smpl" in self.args.representation:
                c["concat_smpl_render"] = smpl_render_latent
                uc["concat_smpl_render"] = smpl_render_latent
            
            if self.model.use_i2v_clip:
                c["image_clip_features"] = image_clip_features
                uc["image_clip_features"] = image_clip_features
            
            progress(0.4, desc="Generating video (this may take a while)...")
            
            # Sample
            sample_func = self.model.sample
            samples_z = sample_func(
                c,
                uc=uc,
                batch_size=1,
                shape=(T, C, H, W),
                ofs=torch.tensor([2.0]).to('cuda'),
                fps=torch.tensor([sampling_fps]).to('cuda'),
            )
            
            progress(0.9, desc="Decoding video...")
            
            # Decode
            samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
            samples_x = self.model.decode_first_stage(samples_z).to(torch.float32)
            samples_x = samples_x.permute(0, 2, 1, 3, 4).contiguous()
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        
        progress(0.95, desc="Saving video...")
        
        # Save video to temp file
        output_path = tempfile.mktemp(suffix=".mp4")
        
        vid = samples[0]  # T C H W
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).numpy().astype(np.uint8)
            gif_frames.append(frame)
        
        with imageio.get_writer(output_path, fps=driving_fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)
        
        progress(1.0, desc="Done!")
        
        return output_path


# ======================= Gradio Interface =======================

# Global model instance
scail_model = SCAILModel()


def generate_video(ref_image, pose_video, prompt, negative_prompt, 
                   sampling_fps, progress=gr.Progress()):
    """Gradio interface function for video generation"""
    
    # Load model if not loaded
    if not scail_model.loaded:
        progress(0, desc="Loading model (this may take a few minutes on first run)...")
        scail_model.load_model()
    
    # Check inputs
    if ref_image is None:
        raise gr.Error("请上传参考图片")
    if pose_video is None:
        raise gr.Error("请上传姿态/驱动视频 (rendered.mp4)")
    
    # Generate
    output_path = scail_model.generate(
        ref_image=ref_image,
        pose_video=pose_video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        sampling_fps=sampling_fps,
        progress=progress
    )
    
    return output_path


# 示例的默认提示词
EXAMPLE_PROMPTS = {
    "001": "一位卷发女性在岩石海岸边欢快地跳舞，穿着时尚的蓝色泳装。她表演各种舞蹈动作，包括旋转、举手，充满活力地享受海边氛围。",
    "002": "一位女性正在表演优美的舞蹈动作，动作流畅自然，充满艺术感。",
}


def load_example(example_name):
    """Load example data"""
    if not example_name:
        raise gr.Error("请先选择一个示例")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(base_dir, "examples", example_name)
    
    # Find reference image
    ref_image_path = None
    for name in ["ref.jpg", "ref.png", "ref_image.jpg", "ref_image.png"]:
        path = os.path.join(example_dir, name)
        if os.path.exists(path):
            ref_image_path = path
            break
    
    # Find pose video (rendered.mp4 has priority, fallback to driving.mp4 for preview)
    pose_video_path = None
    for name in ["rendered_aligned.mp4", "rendered.mp4", "driving.mp4"]:
        path = os.path.join(example_dir, name)
        if os.path.exists(path):
            pose_video_path = path
            # Warn if using driving.mp4
            if name == "driving.mp4":
                gr.Warning(
                    f"正在使用 driving.mp4 作为姿态视频。为获得最佳效果，请使用 SCAIL-Pose 提取姿态生成 rendered.mp4"
                )
            break
    
    if ref_image_path is None:
        raise gr.Error(f"在 {example_dir} 中未找到参考图片")
    if pose_video_path is None:
        raise gr.Error(f"在 {example_dir} 中未找到视频")
    
    # Get default prompt for this example
    default_prompt = EXAMPLE_PROMPTS.get(example_name, "")
    
    return ref_image_path, pose_video_path, default_prompt


# Create Gradio interface
def create_app():
    # Get available examples
    base_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(base_dir, "examples")
    available_examples = []
    if os.path.exists(examples_dir):
        available_examples = [d for d in os.listdir(examples_dir) 
                            if os.path.isdir(os.path.join(examples_dir, d))]
        available_examples.sort()
    
    with gr.Blocks(
        title="SCAIL - 角色动画生成",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .warning-box { 
            background-color: #fff3cd; 
            border: 1px solid #ffc107; 
            padding: 10px; 
            border-radius: 5px;
            margin: 10px 0;
            color: #856404;
        }
        """
    ) as app:
        gr.Markdown(
            """
            # 🎭 SCAIL: 工作室级角色动画生成
            
            通过参考图片和姿态驱动视频生成高质量的角色动画。
            
            **使用说明：**
            1. 上传一张参考图片（你想要动画化的角色）
            2. 上传一个姿态视频（通过 [SCAIL-Pose](https://github.com/teal024/SCAIL-Pose) 渲染的姿态序列）
            3. 输入详细的提示词描述动画内容
            4. 点击"生成动画"开始生成
            
            **提示：**
            - 模型在使用**详细的提示词**描述角色和动作时效果最佳
            - 使用 SCAIL-Pose 从驱动视频中提取姿态可获得最佳效果
            - 首次运行需要加载模型，可能需要较长时间（约140亿参数）
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                ref_image = gr.Image(
                    label="参考图片",
                    type="filepath",
                    height=300,
                    sources=["upload", "clipboard"]
                )
                
                pose_video = gr.Video(
                    label="姿态视频 (SCAIL-Pose 渲染的 rendered.mp4)",
                    height=300
                )
                
                # Examples section
                if available_examples:
                    gr.Markdown("### 📂 加载示例")
                    with gr.Row():
                        example_dropdown = gr.Dropdown(
                            choices=available_examples,
                            label="选择示例",
                            value=None,
                            scale=3
                        )
                        load_example_btn = gr.Button("加载", size="sm", scale=1)
                
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="提示词",
                    placeholder="输入详细的动画描述...\n\n示例：一位卷发女性在岩石海岸边欢快地跳舞，穿着时尚的蓝色泳装。她表演各种舞蹈动作，包括旋转、举手，充满活力地享受海边氛围。",
                    lines=5,
                    value=""
                )
                
                negative_prompt = gr.Textbox(
                    label="负面提示词（可选）",
                    placeholder="需要避免的内容，如：变形的手、模糊、低质量",
                    lines=2,
                    value=""
                )
                
                with gr.Accordion("高级设置", open=False):
                    sampling_fps = gr.Slider(
                        minimum=8,
                        maximum=30,
                        value=16,
                        step=1,
                        label="采样帧率",
                        info="生成视频的每秒帧数"
                    )
                
                generate_btn = gr.Button(
                    "🎬 生成动画", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    <div class="warning-box">
                    ⚠️ <b>注意：</b> 生成过程可能需要几分钟，具体取决于视频长度和GPU性能。
                    </div>
                    """
                )
        
        with gr.Row():
            output_video = gr.Video(
                label="生成的动画",
                height=400
            )
        
        # Load example handler
        if available_examples:
            load_example_btn.click(
                fn=load_example,
                inputs=[example_dropdown],
                outputs=[ref_image, pose_video, prompt]
            )
        
        # Generate handler
        generate_btn.click(
            fn=generate_video,
            inputs=[ref_image, pose_video, prompt, negative_prompt, sampling_fps],
            outputs=[output_video]
        )
        
        gr.Markdown(
            """
            ---
            ### 📚 相关资源
            - [项目主页](https://teal024.github.io/SCAIL/) | [论文](https://arxiv.org/abs/2512.05905) | [GitHub](https://github.com/zai-org/SCAIL)
            - [SCAIL-Pose](https://github.com/teal024/SCAIL-Pose) - 姿态提取和渲染工具
            
            ### 📝 引用
            ```bibtex
            @article{yan2025scail,
              title={SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations},
              author={Yan, Wenhao and Ye, Sheng and Yang, Zhuoyi and Teng, Jiayan and Dong, ZhenHui and Wen, Kairui and Gu, Xiaotao and Liu, Yong-Jin and Tang, Jie},
              journal={arXiv preprint arXiv:2512.05905},
              year={2025}
            }
            ```
            """
        )
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
