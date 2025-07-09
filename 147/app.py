import gradio as gr
import os
import torch
import tempfile
import shutil
import subprocess
from diffsynth import ModelManager, WanVideoPipeline, save_video, load_state_dict
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model
import numpy as np
from diffsynth.models.camera import CamVidEncoder
import cv2
import random
import math
import torch.nn.functional as F
from tqdm import tqdm
from imageio.v3 import imread, imwrite
import sys

# Add utility imports
sys.path.append("DepthCrafter")

try:
    from utils.nv_utils import *
    from utils.dc_utils import read_images_frames, read_video_frames
    from utils.render_utils import get_rays_from_pose
    from utils.depth_utils import DepthCrafterDemo
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"导入工具模块失败: {e}")
    UTILS_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("trimesh 未安装")
    TRIMESH_AVAILABLE = False


def load_video_frames(video_path, num_frames, height, width):
    """Load and process video frames"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < num_frames:
        print(f"Warning: Video has only {frame_count} frames, but {num_frames} required")
    
    # Get frame indices
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        indices = np.arange(frame_count)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # Pad or trim frames to match num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        frames = frames[:num_frames]
    
    frames = np.array(frames)
    video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    return video_tensor

def load_mask_frames(mask_path, num_frames, height, width):
    """Load and process mask frames"""
    cap = cv2.VideoCapture(mask_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < num_frames:
        print(f"Warning: Mask video has only {frame_count} frames, but {num_frames} required")
    
    # Get frame indices
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        indices = np.arange(frame_count)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            # Apply random erosion like in original code
            import random
            kernel = random.randint(2, 8)
            frame = cv2.erode(frame, np.ones((kernel, kernel), np.uint8), iterations=1)
            frames.append(frame)
    
    cap.release()
    
    # Pad or trim frames to match num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        frames = frames[:num_frames]
    
    frames = np.array(frames)
    mask_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    mask_tensor = (mask_tensor / 255.0 > 0.5).float()  # Binarize mask
    return mask_tensor


def generate_mask_from_video(video_path, num_frames, height, width, mask_type="motion"):
    """Generate mask from input video automatically"""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count < num_frames:
            print(f"警告: 视频只有 {frame_count} 帧，但需要 {num_frames} 帧")
        
        # Get frame indices
        if frame_count >= num_frames:
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        else:
            indices = np.arange(frame_count)
        
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (width, height))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < 2:
            # If only one frame, create a partial mask (not full white)
            mask_frame = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray mask instead of white
            masks = [mask_frame] * num_frames
        else:
            # Generate motion-based mask with improved algorithm
            masks = []
            prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            
            for i, frame in enumerate(frames):
                curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if i == 0:
                    # First frame - create partial mask
                    mask = np.ones((height, width), dtype=np.uint8) * 128
                else:
                    # Calculate frame difference with better threshold
                    diff = cv2.absdiff(prev_frame, curr_frame)
                    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)  # Lower threshold
                    
                    # Morphological operations to clean up the mask
                    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Dilate to make mask areas larger
                    mask = cv2.dilate(mask, kernel, iterations=2)
                    
                    # Ensure mask has reasonable content but not too much
                    mask_ratio = cv2.countNonZero(mask) / (height * width)
                    if mask_ratio < 0.02:  # Less than 2% pixels
                        # Create a center mask if motion is too small
                        center_h, center_w = height // 2, width // 2
                        mask = np.zeros((height, width), dtype=np.uint8)
                        mask[center_h-height//4:center_h+height//4, 
                             center_w-width//4:center_w+width//4] = 255
                    elif mask_ratio > 0.8:  # More than 80% pixels
                        # Reduce mask if too large
                        mask = cv2.erode(mask, kernel, iterations=3)
                
                # Convert to 3-channel mask
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                masks.append(mask_3ch)
                prev_frame = curr_frame
        
        # Pad or trim masks to match num_frames
        if len(masks) < num_frames:
            masks += [masks[-1]] * (num_frames - len(masks))
        else:
            masks = masks[:num_frames]
        
        masks = np.array(masks)
        mask_tensor = torch.from_numpy(masks).permute(3, 0, 1, 2).float()
        mask_tensor = (mask_tensor / 255.0 > 0.5).float()  # Binarize mask
        
        # Debug: Print mask statistics
        mask_sum = mask_tensor.sum().item()
        total_pixels = mask_tensor.numel()
        mask_ratio = mask_sum / total_pixels
        print(f"蒙版统计: {mask_ratio:.2%} 的像素被激活")
        
        return mask_tensor
        
    except Exception as e:
        print(f"生成蒙版时出错: {e}")
        # Fallback: create a center mask instead of full white
        mask = np.zeros((3, num_frames, height, width), dtype=np.float32)
        center_h, center_w = height // 2, width // 2
        mask[:, :, center_h-height//4:center_h+height//4, center_w-width//4:center_w+width//4] = 1.0
        return torch.from_numpy(mask)


def add_lora_to_model(model, lora_rank=16, lora_alpha=16.0, lora_target_modules="q,k,v,o,ffn.0,ffn.2", 
                     init_lora_weights="kaiming", pretrained_path=None, state_dict_converter=None):
    """Add LoRA to model"""
    if init_lora_weights == "kaiming":
        init_lora_weights = True
        
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    if pretrained_path is not None:
        state_dict = load_state_dict(pretrained_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(f"LORA: {num_updated_keys} parameters loaded from {pretrained_path}. {num_unexpected_keys} unexpected.")


class EX4DPipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipe = None
        self.model_loaded = False
        self.depth_estimator = None
        self.glctx = None
        
        # Default model paths
        self.ex4d_path = 'models/EX-4D/ex4d.ckpt'
        self.text_encoder_path = 'models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth'
        self.vae_path = 'models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth'
        self.clip_path = 'models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
        self.dit_dir = 'models/Wan-AI/Wan2.1-I2V-14B-480P/'
        
    def load_models(self):
        """Load all models"""
        if self.model_loaded:
            return "模型已加载"
            
        try:
            print("正在加载模型...")
            dit_paths = [
                os.path.join(self.dit_dir, f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors")
                for i in range(1, 8)
            ]
            
            # Check if model files exist
            missing_files = []
            for path in [self.ex4d_path, self.text_encoder_path, self.vae_path, self.clip_path] + dit_paths:
                if not os.path.exists(path):
                    missing_files.append(path)
            
            if missing_files:
                return f"缺少模型文件: {missing_files[:3]}..."  # Show first 3 missing files
            
            model_manager = ModelManager(torch_dtype=torch.bfloat16, device=self.device)
            model_manager.load_models([dit_paths, self.text_encoder_path, self.vae_path, self.clip_path])
            
            self.pipe = WanVideoPipeline.from_model_manager(model_manager, device=self.device)
            self.pipe.camera_encoder = CamVidEncoder(16, 1024, 5120).to(self.device, dtype=torch.bfloat16)
            
            # Add LoRA
            add_lora_to_model(self.pipe.denoising_model(), pretrained_path=self.ex4d_path, lora_rank=16, lora_alpha=16.0)
            self.pipe.load_cam(self.ex4d_path)
            self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
            self.pipe.camera_encoder = self.pipe.camera_encoder.to(self.device, dtype=torch.bfloat16)
            self.pipe.camera_encoder.eval()
            
            self.model_loaded = True
            return "模型加载成功！"
            
        except Exception as e:
            return f"模型加载错误: {str(e)}"
    
    def generate_mask_video_file(self, input_video, num_frames, height, width):
        """Generate and save mask video as a file for preview"""
        try:
            mask_tensor = generate_mask_from_video(input_video, num_frames, height, width)
            
            # Convert tensor back to video format for saving
            mask_np = mask_tensor.permute(1, 2, 3, 0).numpy()  # (frames, height, width, channels)
            mask_np = (mask_np * 255).astype(np.uint8)
            
            # Save mask video to temporary file
            mask_video_path = tempfile.mktemp(suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(mask_video_path, fourcc, 15.0, (width, height))
            
            for frame in mask_np:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            return mask_video_path
            
        except Exception as e:
            print(f"生成蒙版视频文件时出错: {e}")
            return None
    
    def reconstruct_video(self, input_video, cam_angle=180, height=512, width=512, num_frames=49):
        """内部重建功能，直接调用深度估计和渲染"""
        if input_video is None:
            return None, None, "请上传输入视频！"
        
        try:
            print(f"开始重建视频: {input_video}")
            print(f"参数: cam_angle={cam_angle}, num_frames={num_frames}, 分辨率: {width}x{height}")
            
            # 加载深度估计模型
            self.load_depth_estimator()
            
            if self.depth_estimator is None or self.glctx is None:
                print("深度估计模型或渲染上下文加载失败，使用简化方法")
                return self.process_input_video_fallback(input_video, cam_angle, height, width, num_frames)
            
            # 创建临时输出目录
            temp_output_dir = tempfile.mkdtemp()
            print(f"创建临时目录: {temp_output_dir}")
            
            try:
                # 直接运行重建流程，传递目标分辨率
                return self.run_render_internal(input_video, temp_output_dir, cam_angle, num_frames, height, width)
                
            except Exception as e:
                print(f"内部重建失败，使用简化方法: {e}")
                return self.process_input_video_fallback(input_video, cam_angle, height, width, num_frames)
            finally:
                # 清理临时目录
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"重建失败，使用简化方法: {e}")
            return self.process_input_video_fallback(input_video, cam_angle, height, width, num_frames)
    
    def run_render_internal(self, video_path, output_dir, cam_angle, num_frames, target_height=512, target_width=512):
        """内部渲染流程，移植自 recon.py"""
        if not UTILS_AVAILABLE:
            raise Exception("工具模块不可用，无法运行内部渲染")
            
        print(f"处理视频: {video_path}")
        print(f"目标分辨率: {target_width}x{target_height}")
        
        # 1. 加载视频帧并调整到目标分辨率
        try:
            frames, _ = read_video_frames(video_path, process_length=num_frames, max_res=max(target_height, target_width))
            # 调整到目标分辨率
            if frames.shape[1] != target_height or frames.shape[2] != target_width:
                resized_frames = []
                for frame in frames:
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    resized_frames.append(resized_frame)
                frames = np.array(resized_frames)
                print(f"视频帧已调整到目标分辨率: {frames.shape}")
        except:
            # 如果读取失败，尝试用 cv2
            cap = cv2.VideoCapture(video_path)
            frame_list = []
            while len(frame_list) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 调整到目标分辨率
                frame = cv2.resize(frame, (target_width, target_height))
                frame_list.append(frame)
            cap.release()
            frames = np.array(frame_list)
            print(f"使用cv2加载并调整分辨率: {frames.shape}")
        
        if frames.shape[0] < num_frames:
            print(f"视频帧数不足: {frames.shape[0]} < {num_frames}")
            # 复制最后一帧补齐
            last_frame = frames[-1:] if len(frames) > 0 else np.zeros((1, 512, 512, 3), dtype=np.uint8)
            frames = np.concatenate([frames] + [last_frame] * (num_frames - frames.shape[0]))
        
        frames = frames[:num_frames]
        
        # 2. 运行深度估计
        print("运行深度估计...")
        try:
            depth_src, intrinsics = self.run_depth_crafter(frames.astype(np.float32) / 255.)
        except Exception as e:
            print(f"深度估计失败: {e}")
            raise e
        
        frames = torch.from_numpy(frames).float().to(self.device)
        depth_src = depth_src.to(self.device)
        
        # 3. 渲染网格
        print("开始渲染...")
        with torch.no_grad():
            old_depth_src = depth_src
            depth_src = depth_src.clone()
            
            # 边界处理
            depth_src[:, :, 0, :] = 100
            depth_src[:, :, -1, :] = 100
            depth_src[:, :, :, 0] = 100
            depth_src[:, :, :, -1] = 100
            
            depth_src = depth_src.unsqueeze(-1)
            rgbs_src = frames
            
            H, W, C = rgbs_src[0].shape
            fov_y = 2 * math.atan2(H, 2 * intrinsics[1, 1])
            fov_x = 2 * math.atan2(W, 2 * intrinsics[0, 0])
            
            # 创建相机内参矩阵
            fx = fy = 0.5 * H / math.tan(fov_y / 2)
            K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], 
                           dtype=torch.float32, device=self.device)
            pose = torch.eye(4, device=self.device)
            ro_src, rd_src = get_rays_from_pose(pose, K, H, W)
            proj = getprojection(fov_x, fov_y, n=1e-3, f=1e3, device=self.device)
            
            # 相机轨迹
            depth_min = depth_src[0].min().item() + 0.15
            camera_poses = self.random_camera_traj(num_frames, depth_src, str(cam_angle), num_frames, depth_min)
            
            video = []
            for idx, poses in enumerate(tqdm(camera_poses, desc="渲染帧")):
                pts_color = rgbs_src[idx]
                pts_xyz = depth_src[idx] * rd_src + ro_src
                faces = self.generate_faces(H, W, self.device)
                
                vertices, new_faces, colors, _ = self.point_to_mesh_cuda(
                    pts_xyz, pts_color, faces, old_depth_src[idx], filter_type="angle")
                
                try:
                    img = self.render_nvdiffrast(vertices, new_faces, colors, proj, poses, fov_x, fov_y, H, W)[0]
                except Exception as e:
                    print(f"渲染第 {idx} 帧失败: {e}")
                    # 使用原始图像作为备用
                    img = torch.cat([pts_color, torch.ones_like(pts_color[:,:,:1]) * 255], dim=-1)
                
                if idx == 0:
                    img[..., :3] = pts_color
                    img[..., 3:] = 255
                else:
                    mask = img[..., 3:]
                    mask[mask > 125] = 255
                    mask[mask <= 125] = 0
                    img[..., 3:] = mask
                    img[..., :3] = img[..., :3] * (mask / 255)
                
                video.append(img.cpu().numpy().astype(np.uint8))
            
            # 保存视频
            os.makedirs(output_dir, exist_ok=True)
            
            cond_path = os.path.join(output_dir, f"color_{cam_angle}.mp4")
            mask_path = os.path.join(output_dir, f"mask_{cam_angle}.mp4")
            
            video = np.stack(video, axis=0).astype(np.uint8)
            
            # 保存彩色和蒙版视频
            imwrite(cond_path, video[..., :3], fps=30)
            imwrite(mask_path, video[..., 3:], fps=30)
            
            print(f"重建完成: {cond_path}, {mask_path}")
            
            # 复制到临时文件供 Gradio 使用
            final_color_path = tempfile.mktemp(suffix='.mp4')
            final_mask_path = tempfile.mktemp(suffix='.mp4')
            
            shutil.copy2(cond_path, final_color_path)
            shutil.copy2(mask_path, final_mask_path)
            
            return final_color_path, final_mask_path, f"重建成功！使用相机角度: {cam_angle}°"
    
    def process_input_video_fallback(self, input_video, cam_angle=180, height=512, width=512, num_frames=49):
        """回退方法：简单的蒙版生成（当真正的重建失败时使用）"""
        try:
            print(f"使用简化方法，目标分辨率: {width}x{height}")
            
            # Generate mask video file for preview (using cam_angle parameter)
            mask_video_path = self.generate_mask_video_file(input_video, num_frames, height, width)
            
            # Resize input video to match target resolution
            color_video_path = self.resize_video_to_target(input_video, height, width, num_frames)
            
            return color_video_path, mask_video_path, f"使用简化方法处理成功！相机角度: {cam_angle}°"
            
        except Exception as e:
            return None, None, f"处理错误: {str(e)}"
    
    def resize_video_to_target(self, input_video, target_height, target_width, num_frames):
        """将输入视频调整到目标分辨率"""
        try:
            cap = cv2.VideoCapture(input_video)
            
            # Create temporary output video
            output_path = tempfile.mktemp(suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 15.0, (target_width, target_height))
            
            frame_count = 0
            while frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    if frame_count == 0:
                        # If no frames read, create a black frame
                        frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    else:
                        # Use last frame if video is shorter
                        break
                
                # Resize frame to target resolution
                frame_resized = cv2.resize(frame, (target_width, target_height))
                out.write(frame_resized)
                frame_count += 1
            
            # Pad with last frame if needed
            if frame_count < num_frames and frame_count > 0:
                for _ in range(num_frames - frame_count):
                    out.write(frame_resized)
            
            cap.release()
            out.release()
            
            print(f"视频已调整到目标分辨率: {target_width}x{target_height}")
            return output_path
            
        except Exception as e:
            print(f"调整视频分辨率失败: {e}")
            return input_video  # Return original if resize fails
    
    def generate_video(self, color_video, mask_video, height=512, width=512, num_frames=49, 
                      num_inference_steps=25, seed=0, progress=gr.Progress()):
        """Generate video using EX-4D"""
        if color_video is None or mask_video is None:
            return None, "请上传彩色视频和蒙版视频！"
        
        # Auto-load models if not loaded
        if not self.model_loaded:
            progress(0.05, desc="正在加载模型...")
            load_result = self.load_models()
            if "错误" in load_result or "缺少" in load_result:
                return None, f"模型加载失败: {load_result}"
        
        try:
            progress(0.2, desc="正在处理输入视频...")
            
            print(f"加载视频，目标分辨率: {width}x{height}, 帧数: {num_frames}")
            
            # Load mask and color videos with specified resolution
            mask_tensor = load_mask_frames(mask_video, num_frames, height, width)
            color_tensor = load_video_frames(color_video, num_frames, height, width)
            
            print(f"彩色视频张量尺寸: {color_tensor.shape}")
            print(f"蒙版视频张量尺寸: {mask_tensor.shape}")
            
            # Ensure both tensors have the same dimensions
            if color_tensor.shape != mask_tensor.shape:
                print(f"警告: 彩色视频和蒙版视频尺寸不匹配，将调整蒙版尺寸")
                # Resize mask to match color video
                mask_np = mask_tensor.permute(1, 2, 3, 0).numpy()  # (frames, height, width, channels)
                target_frames, target_channels, target_height, target_width = color_tensor.shape
                
                resized_mask_frames = []
                for i in range(target_frames):
                    if i < mask_np.shape[0]:
                        mask_frame = mask_np[i]
                    else:
                        mask_frame = mask_np[-1]  # Use last frame if not enough frames
                    
                    # Resize mask frame
                    mask_frame_resized = cv2.resize(mask_frame, (target_width, target_height))
                    if len(mask_frame_resized.shape) == 2:
                        mask_frame_resized = np.stack([mask_frame_resized] * target_channels, axis=-1)
                    resized_mask_frames.append(mask_frame_resized)
                
                resized_mask_np = np.array(resized_mask_frames)
                mask_tensor = torch.from_numpy(resized_mask_np).permute(3, 0, 1, 2).float()
                mask_tensor = (mask_tensor / 255.0 > 0.5).float()  # Binarize mask
                print(f"调整后蒙版视频张量尺寸: {mask_tensor.shape}")
            
            color_tensor = (color_tensor * mask_tensor).to(torch.bfloat16) * 2 - 1
            mask_tensor = mask_tensor.to(torch.bfloat16) * 2 - 1
            
            progress(0.4, desc="设置提示词...")
            
            # Set prompts
            prompt = "4K ultra HD, surround motion, realistic tone, panoramic shot, wide-angle view, cinematic quality"
            negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
            
            progress(0.6, desc="正在运行推理...")
            
            # Run inference
            with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type=self.device):
                input_cond = color_tensor.to(self.device)[None]  # Add batch dimension
                input_mask = mask_tensor.to(self.device)[None]
                
                output_video = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    video=input_cond,
                    mask=input_mask,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    tiled=False,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                )
            
            progress(0.9, desc="正在保存输出视频...")
            
            # Save output video to temporary file
            output_path = tempfile.mktemp(suffix='.mp4')
            save_video(output_video, output_path, fps=15, quality=8)
            
            progress(1.0, desc="完成!")
            return output_path, "视频生成成功！"
            
        except Exception as e:
            return None, f"生成过程中出错: {str(e)}"
    
    def load_depth_estimator(self):
        """Load depth estimation model"""
        if not UTILS_AVAILABLE:
            print("工具模块不可用，无法加载深度估计器")
            return
            
        if self.depth_estimator is None:
            try:
                self.depth_estimator = DepthCrafterDemo(
                    unet_path="models/DepthCrafter",
                    pre_train_path="models/stable-video-diffusion-img2vid",
                    cpu_offload=None,
                    device=self.device
                )
                print("深度估计模型加载成功")
            except Exception as e:
                print(f"加载深度估计模型失败: {e}")
                self.depth_estimator = None
        
        if self.glctx is None:
            try:
                import nvdiffrast.torch as dr
                self.glctx = dr.RasterizeCudaContext(device=self.device)
                print("渲染上下文创建成功")
            except ImportError as e:
                print(f"nvdiffrast 未安装: {e}")
                self.glctx = None
            except Exception as e:
                print(f"创建渲染上下文失败: {e}")
                self.glctx = None
    
    def point_to_mesh_cuda(self, pts, rgbs, faces, old_depth_src, min_angle_deg=2.5, filter_type="angle"):
        """Convert point cloud to mesh with filtering"""
        h, w = rgbs.shape[:2]
        vertices = pts.reshape(-1, 3)
        masks = torch.ones((h, w, 1), dtype=torch.uint8).to(rgbs.device) * 255
        rgbs = torch.cat([rgbs, masks], axis=-1)
        colors = rgbs.reshape(-1, 4)
        
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0), dim=1)
        
        if filter_type == "angle":
            def angle_between(v1, v2):
                cos_theta = torch.sum(v1 * v2, -1) / (
                    torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12
                )
                return torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

            a = angle_between(v1 - v0, v2 - v0)
            b = angle_between(v2 - v1, v0 - v1)
            c = angle_between(v0 - v2, v1 - v2)
            min_angles = torch.minimum(torch.minimum(a, b), c)

            valid_faces = min_angles >= min_angle_deg
            z_range = vertices[:, 2].max() - vertices[:, 2].min()

            z01, z12, z20 = torch.abs((v0 - v1)[:, 2]), torch.abs((v1 - v2)[:, 2]), torch.abs((v2 - v0)[:, 2])
            y01, y12, y20 = torch.abs((v0 - v1)[:, 1]), torch.abs((v1 - v2)[:, 1]), torch.abs((v2 - v0)[:, 1])
            x01, x12, x20 = torch.abs((v0 - v1)[:, 0]), torch.abs((v1 - v2)[:, 0]), torch.abs((v2 - v0)[:, 0])
            z_max = torch.maximum(torch.maximum(z01, z12), z20)
            y_max = torch.maximum(torch.maximum(y01, y12), y20)
            x_max = torch.maximum(torch.maximum(x01, x12), x20)
            proj_max = torch.maximum(torch.maximum(x_max, y_max), z_max)
            valid_faces2 = (proj_max / z_range < 0.013)
            valid_faces_final = valid_faces & valid_faces2
            faces = faces[valid_faces_final]
        
        return vertices, faces, colors, None
    
    def render_nvdiffrast(self, vertices, faces, colors, proj, poses, fovx, fovy, h, w, near=1e-3, far=1e3):
        """Render mesh using nvdiffrast"""
        try:
            import nvdiffrast.torch as dr
        except ImportError:
            raise ImportError("nvdiffrast 未安装，无法进行渲染")
        
        def transform_pos(mtx, pos):
            t_mtx = torch.from_numpy(mtx).to(pos.device) if isinstance(mtx, np.ndarray) else mtx
            posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
            return torch.matmul(posw, t_mtx.t())[None, ...]

        def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, h, w):
            pos_clip = transform_pos(mtx, pos)
            rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[h, w])
            color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
            color = dr.antialias(color, rast_out, pos_clip, pos_idx)
            return color
            
        poses_copy = poses.clone()
        poses_copy[0,:] *= -1
        poses_copy[1,:] *= -1
        poses_copy[2,:] *= -1
        mvp = proj @ poses_copy
        return render(self.glctx, mvp, vertices, faces, colors, faces, h, w)
    
    def get_camera_pose(self, eye, center):
        """Calculate camera pose matrix"""
        up = np.array((0, 1, 0), dtype=np.float32)
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                return v
            return v / norm
            
        forward = normalize(center - eye)
        right = normalize(np.cross(forward, up))
        new_up = normalize(np.cross(right, forward))
        view = np.zeros((4, 4), dtype=np.float32)
        view[0, 0:3] = right
        view[1, 0:3] = new_up
        view[2, 0:3] = forward
        view[0:3, 3] = -np.array([np.dot(right, eye), np.dot(new_up, eye), np.dot(forward, eye)])
        view[3, 3] = 1.0
        return torch.from_numpy(view)
    
    def random_camera_traj(self, n_frames, depth_src, random_type, num_frames, depth_min):
        """Generate random camera trajectory"""
        rounds = (49 / num_frames) if 49 > num_frames else num_frames / 49
        
        if random_type == "180":
            radius = depth_min
            eyes = np.zeros((num_frames, 3))
            angle = np.linspace(0, 2 * rounds * np.pi, num_frames)
            eyes[:, 0] = np.sin(angle) * radius
            eyes[:, 1] = (np.cos(angle) - 1) * radius * 0.2
            centers = np.zeros((num_frames, 3))
            centers[:, 2] = radius
        else:
            ta = int(random_type)
            angle = np.linspace(0, (ta / 180 * np.pi) * rounds, num_frames)
            eyes = np.zeros((num_frames, 3))
            radius = depth_min
            eyes[:, 0] = np.sin(angle) * radius
            eyes[:, 1] = (np.abs(np.cos(angle)) - 1) * radius * 0.2
            eyes[:, 2] = radius - radius * np.abs(np.cos(angle))
            centers = np.zeros((num_frames, 3))
            centers[:, -1] = radius
            
        camera_poses = torch.stack([self.get_camera_pose(eye, center) for eye, center in zip(eyes, centers)], 0)
        camera_poses = camera_poses.to(depth_src.device)
        return camera_poses
    
    def generate_faces(self, H, W, device):
        """Generate mesh faces for the grid"""
        idx = np.arange(H * W).reshape(H, W)
        faces = torch.from_numpy(np.concatenate([
                np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], axis=-1),
                np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()], axis=-1)
            ], axis=0)).int().to(device)
        faces = faces[:,[1,0,2]]
        faces = torch.cat([
                faces,
                torch.tensor([[W - 1, 0, H * W - 1]]).int().to(device),
                torch.tensor([[H * W - 1, 0, (H - 1) * W]]).int().to(device),
            ], 0)
        return faces
    
    def run_depth_crafter(self, frames, near=0.0001, far=10000, 
                         depth_inference_steps=5, depth_guidance_scale=1.0, 
                         window_size=110, overlap=25):
        """Run depth estimation using DepthCrafter"""
        depths, _ = self.depth_estimator.infer(
            frames, near, far, depth_inference_steps, depth_guidance_scale,
            window_size=window_size, overlap=overlap
        )
        f = 500
        cx = depths.shape[-1] // 2
        cy = depths.shape[-2] // 2
        intrinsics = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        return depths, intrinsics


# Initialize pipeline
pipeline = EX4DPipeline()


def create_interface():
    """Create Gradio interface"""
    
    # Default settings (hidden from user, automatically managed)
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512
    
    with gr.Blocks(title="EX-4D 视频生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# EX-4D 视频生成 WebUI")
        gr.Markdown("基于 EX-4D 模型的两步视频生成流程：第一步重建彩色和蒙版视频，第二步生成最终增强视频。")
        gr.Markdown("🎯 **智能分辨率适配**：支持任意分辨率视频上传，系统自动优化处理，无需手动设置！")
        
        # Hidden components for internal use
        height = gr.State(value=DEFAULT_HEIGHT)
        width = gr.State(value=DEFAULT_WIDTH)
        
        with gr.Row():
            # Left panel - Step 1: Reconstruction
            with gr.Column(scale=1):
                gr.Markdown("## 第一步：视频重建")
                gr.Markdown("上传任意分辨率的输入视频，系统自动生成彩色视频和蒙版视频")
                
                input_video = gr.Video(label="📁 输入视频（支持任意分辨率）", height=200)
                cam_angle = gr.Dropdown(
                    choices=[30, 60, 90, 180],
                    value=180,
                    label="📷 相机角度",
                    info="控制视野范围：30°=窄视野，180°=广视野"
                )
                
                reconstruct_btn = gr.Button("🔍 开始重建视频", variant="secondary", size="lg")
                
                gr.Markdown("### 重建结果预览")
                with gr.Row():
                    color_video_preview = gr.Video(label="生成的彩色视频", height=150)
                    mask_video_preview = gr.Video(label="生成的蒙版视频", height=150)
                
                reconstruct_status = gr.Textbox(label="重建状态", interactive=False)
                
            # Right panel - Step 2: Generation  
            with gr.Column(scale=1):
                gr.Markdown("## 第二步：视频生成")
                gr.Markdown("使用重建的彩色和蒙版视频生成最终增强视频")
                
                gr.Markdown("### 输入视频（从第一步自动填充）")
                color_video = gr.Video(label="彩色视频", height=150, interactive=True)
                mask_video = gr.Video(label="蒙版视频", height=150, interactive=True)
                
                gr.Markdown("### 生成参数")
                with gr.Row():
                    num_frames = gr.Slider(1, 100, value=49, step=1, label="🎬 帧数")
                    num_inference_steps = gr.Slider(10, 50, value=25, step=1, label="⚙️ 推理步数")
                
                seed = gr.Number(value=0, label="🎲 随机种子", precision=0)
                
                generate_btn = gr.Button("🚀 生成最终视频", variant="primary", size="lg")
                
                gr.Markdown("### 最终输出")
                output_video = gr.Video(label="生成的视频", height=300)
                generation_status = gr.Textbox(label="生成状态", interactive=False)
                
                gr.Markdown("## 📖 使用说明")
                gr.Markdown("""
                **🎯 完全自动化流程：**
                1. **只需上传**：上传任意分辨率的视频（720p、1080p、4K等都支持）
                2. **自动处理**：系统自动调整分辨率，保证流程兼容性
                3. **一键重建**：选择相机角度，点击"开始重建视频" 
                4. **一键生成**：调整参数，点击"生成最终视频"
                
                **🔧 高级用法：**
                - 您也可以直接在右侧上传自己的彩色视频和蒙版视频
                - 或使用下方示例快速开始
                
                **💡 智能特性：** 
                - ✅ **分辨率自适应**：无需关心输入视频分辨率
                - ✅ **自动对齐**：彩色视频和蒙版视频分辨率自动匹配
                - ✅ **相机角度**：30°-180°，角度越大视野越广
                - ✅ **蒙版含义**：白色=修改区域，黑色=保持原样
                """)
        
        # Examples section
        gr.Markdown("## 🎮 示例")
        gr.Markdown("点击下方示例快速体验：")
        
        # Two sets of examples: one for input, one for color+mask
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 示例1：从输入视频开始")
                examples_input = gr.Examples(
                    examples=[
                        [
                            "examples/flower/input.mp4",  # input_video
                            180,  # cam_angle
                            49,   # num_frames
                            25,   # num_inference_steps
                            0     # seed
                        ]
                    ],
                    inputs=[
                        input_video, cam_angle, num_frames, num_inference_steps, seed
                    ],
                    label="花朵视频示例"
                )
            
            with gr.Column():
                gr.Markdown("### 示例2：直接使用重建结果")
                examples_output = gr.Examples(
                    examples=[
                        [
                            "examples/flower/render_180.mp4",  # color_video
                            "examples/flower/mask_180.mp4",    # mask_video
                            49,   # num_frames
                            25,   # num_inference_steps
                            0     # seed
                        ]
                    ],
                    inputs=[
                        color_video, mask_video, num_frames, num_inference_steps, seed
                    ],
                    label="彩色+蒙版视频示例"
                )
        
        # Event handlers
        def reconstruct_and_auto_fill(input_video, cam_angle, num_frames):
            """Reconstruct videos and auto-fill generation inputs"""
            # Use default resolution (hidden from user)
            height_val = DEFAULT_HEIGHT
            width_val = DEFAULT_WIDTH
            
            # First, run the reconstruction
            color_preview, mask_preview, status = pipeline.reconstruct_video(
                input_video, cam_angle, height_val, width_val, num_frames
            )
            
            # Then auto-fill the generation inputs
            if color_preview and mask_preview and "成功" in status:
                fill_status = "第一步完成！已自动填充第二步的输入视频，分辨率已优化"
                return color_preview, mask_preview, status, color_preview, mask_preview, fill_status
            else:
                return color_preview, mask_preview, status, None, None, "第一步失败，无法自动填充"
        
        def generate_with_defaults(color_video, mask_video, num_frames, num_inference_steps, seed):
            """Generate video with default resolution"""
            height_val = DEFAULT_HEIGHT
            width_val = DEFAULT_WIDTH
            return pipeline.generate_video(
                color_video, mask_video, height_val, width_val, num_frames, 
                num_inference_steps, seed
            )
        
        reconstruct_btn.click(
            fn=reconstruct_and_auto_fill,
            inputs=[input_video, cam_angle, num_frames],
            outputs=[color_video_preview, mask_video_preview, reconstruct_status, 
                    color_video, mask_video, generation_status]
        )
        
        generate_btn.click(
            fn=generate_with_defaults,
            inputs=[
                color_video, mask_video, num_frames, 
                num_inference_steps, seed
            ],
            outputs=[output_video, generation_status]
        )
    
    return demo


if __name__ == "__main__":
    # Launch configuration
    HOST = "0.0.0.0"
    PORT = 7860
    SHARE = False  # Set to True to create public link
    
    demo = create_interface()
    demo.launch(
        server_name=HOST,
        server_port=PORT,
        share=SHARE,
        show_api=True,
        debug=False
    )