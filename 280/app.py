"""
EgoX Gradio Web 应用程序
基于 Wan2.1 的第一人称视角视频生成 Demo
"""

import os
import json
import random
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch

# 全局变量存储 pipeline
PIPE = None
MODEL_LOADED = False

# 默认路径配置
DEFAULT_MODEL_PATH = "./checkpoints/Wan2.1-I2V-14B-480P-Diffusers"
DEFAULT_LORA_PATH = "./checkpoints/EgoX/pytorch_lora_weights.safetensors"
ITW_META_PATH = "./example/in_the_wild/meta.json"
EGO4D_META_PATH = "./example/egoexo4D/meta.json"


def set_seed(seed: Optional[int]) -> None:
    """设置随机种子以确保可重复性"""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_path: str, lora_path: str):
    """加载模型"""
    global PIPE, MODEL_LOADED
    
    if MODEL_LOADED:
        return "✅ 模型已加载"
    
    try:
        from transformers import CLIPVisionModel
        from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA as WanTransformer3DModel
        from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline
        
        dtype = torch.bfloat16
        transformer_path = os.path.join(model_path, 'transformer')
        
        print("正在加载 Transformer...")
        transformer = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype)
        
        print("正在加载 Image Encoder...")
        image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)
        
        print("正在创建 Pipeline...")
        PIPE = WanWidthConcatImageToVideoPipeline.from_pretrained(
            model_path, 
            image_encoder=image_encoder, 
            transformer=transformer, 
            torch_dtype=dtype
        )
        
        if lora_path and os.path.exists(lora_path):
            print("正在加载 LoRA 权重...")
            PIPE.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
            PIPE.fuse_lora(components=["transformer"], lora_scale=1.0)
        
        PIPE.to("cuda")
        MODEL_LOADED = True
        
        return "✅ 模型加载成功！"
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"


def load_meta_data(meta_file: str):
    """加载元数据文件"""
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    return meta_data['test_datasets']


def get_example_choices(meta_file: str):
    """获取示例选项列表"""
    try:
        meta_data = load_meta_data(meta_file)
        choices = []
        for i, meta in enumerate(meta_data):
            exo_path = meta['exo_path']
            take_name = exo_path.split('/')[-2]
            choices.append(f"{i}: {take_name}")
        return choices
    except:
        return []


def compute_gga_attention(meta, is_in_the_wild: bool = False):
    """计算 GGA 注意力图"""
    from core.finetune.datasets.utils import iproj_disp
    
    device = 'cpu'
    C, F, H, W = 16, 13, 56, 154
    exo_H, exo_W = H, W - H
    W = H
    
    # 加载深度图
    take_name = meta['exo_path'].split('/')[-2]
    depth_root = "/".join(meta['exo_path'].split('/')[:3])
    depth_map_path = Path(os.path.join(depth_root, 'depth_maps', take_name))
    
    depth_maps = []
    for depth_map_file in sorted(depth_map_path.glob("*.npy")):
        depth_map = np.load(depth_map_file)
        depth_maps.append(torch.from_numpy(depth_map).unsqueeze(0))
    depth_maps = torch.cat(depth_maps, dim=0)
    
    # 获取相机参数
    ego_intrinsic = torch.tensor(meta['ego_intrinsics'])
    ego_extrinsic = torch.tensor(meta['ego_extrinsics'])
    camera_extrinsic = torch.tensor(meta['camera_extrinsics'])
    camera_intrinsic = torch.tensor(meta['camera_intrinsics'])
    
    # 处理外参矩阵
    if ego_extrinsic.shape[1] == 3 and ego_extrinsic.shape[2] == 4:
        ego_extrinsic = torch.cat([
            ego_extrinsic, 
            torch.tensor([[[0, 0, 0, 1]]], dtype=ego_extrinsic.dtype).expand(ego_extrinsic.shape[0], -1, -1)
        ], dim=1)
    if camera_extrinsic.shape == (3, 4):
        camera_extrinsic = torch.cat([
            torch.tensor(camera_extrinsic, dtype=ego_extrinsic.dtype), 
            torch.tensor([[0, 0, 0, 1]], dtype=ego_extrinsic.dtype)
        ], dim=0)
    
    # 缩放内参
    scale = 1/8
    scaled_intrinsic = ego_intrinsic.clone()
    scaled_intrinsic[0, 0] *= scale
    scaled_intrinsic[1, 1] *= scale
    scaled_intrinsic[0, 2] *= scale
    scaled_intrinsic[1, 2] *= scale
    
    # 创建像素坐标
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    ones = torch.ones_like(xs)
    pixel_coords = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).to(dtype=ego_intrinsic.dtype)
    
    pixel_coords_cv = pixel_coords[..., :2].cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
    K = scaled_intrinsic.cpu().numpy().astype(np.float32)
    
    # Ego cam 畸变系数 (Project Aria) 
    distortion_coeffs = np.array([[-0.02340373583137989, 0.09388021379709244, -0.06088035926222801, 
                                   0.0053304750472307205, 0.003342868760228157, -0.0006356257363222539,
                                   0.0005087381578050554, -0.0004747129278257489, -0.0011330085108056664,
                                   -0.00025734835071489215, 0.00009328465239377692, 0.00009424977179151028]])
    D = distortion_coeffs.astype(np.float32)
    normalized_points = cv2.undistortPoints(pixel_coords_cv, K, D, R=np.eye(3), P=np.eye(3))
    
    normalized_points = torch.from_numpy(normalized_points).squeeze(1).to(device)
    ones = torch.ones_like(normalized_points[..., :1])
    cam_rays_fish = torch.cat([normalized_points, ones], dim=-1)
    cam_rays = cam_rays_fish / torch.norm(cam_rays_fish, dim=-1, keepdim=True)
    cam_rays = cam_rays @ ego_extrinsic[::4, :3, :3]
    cam_rays = cam_rays.view(F, H, W, 3)
    
    # 处理相机内参
    height, width = depth_maps.shape[1], depth_maps.shape[2]
    cx = width / 2.0
    cy = height / 2.0
    camera_intrinsic_scale_y = cy / camera_intrinsic[1, 2]
    camera_intrinsic_scale_x = cx / camera_intrinsic[0, 2]
    camera_intrinsic[0, 0] = camera_intrinsic[0, 0] * camera_intrinsic_scale_x
    camera_intrinsic[1, 1] = camera_intrinsic[1, 1] * camera_intrinsic_scale_y
    camera_intrinsic[0, 2] = cx
    camera_intrinsic[1, 2] = cy
    
    camera_intrinsic_array = np.array([camera_intrinsic[0, 0], camera_intrinsic[1, 1], cx, cy])
    
    disp_v, disp_u = torch.meshgrid(
        torch.arange(depth_maps.shape[1], device=device).float(),
        torch.arange(depth_maps.shape[2], device=device).float(),
        indexing="ij",
    )
    
    disp = torch.ones_like(disp_v)
    pts, _, _ = iproj_disp(torch.from_numpy(camera_intrinsic_array), disp.cpu(), disp_u.cpu(), disp_v.cpu())
    
    if isinstance(pts, torch.Tensor):
        pts = pts.to(device)
    else:
        pts = torch.from_numpy(pts).to(device).float()
    
    rays = pts[..., :3]
    rays = rays / rays[..., 2:3]
    rays = rays.unsqueeze(0).expand(depth_maps.size(0), -1, -1, -1)
    camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)
    
    pcd_camera = rays * depth_maps.unsqueeze(-1)
    point_map = pcd_camera.to(dtype=camera_extrinsics_c2w.dtype)
    point_map = torch.tensor(point_map)
    
    p_f, p_h, p_w, p_p = point_map.shape
    point_map_world = point_map.reshape(-1, 3)
    
    camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)
    ones_point = torch.ones(point_map_world.shape[0], 1, device=point_map_world.device)
    point_map_world = torch.cat([point_map_world, ones_point], dim=-1)
    point_map_world = (camera_extrinsics_c2w @ point_map_world.T).T[..., :3]
    point_map = point_map_world.reshape(p_f, p_h, p_w, 3).permute(0, 3, 1, 2)
    
    point_map = point_map[:, :, (point_map.shape[2] - 448)//2:(point_map.shape[2] + 448)//2, 
                          (point_map.shape[3] - 784)//2:(point_map.shape[3] + 784)//2]
    point_map = torch.nn.functional.interpolate(point_map, size=(exo_H, exo_W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
    ego_extrinsic_c2w = torch.linalg.inv(ego_extrinsic)
    cam_origins = ego_extrinsic_c2w[::4, :3, 3].unsqueeze(1).expand(-1, exo_H * exo_W, -1)
    cam_origins = cam_origins.view(F, exo_H, exo_W, 3)
    
    if point_map.size(0) != ego_extrinsic_c2w.size(0):
        min_size = min(point_map.size(0), ego_extrinsic_c2w.size(0))
        point_map = point_map[:min_size]
    
    point_vecs_per_frame = []
    for j in range(cam_origins.size(0)):
        point_vec = point_map[::4] - cam_origins[j].unsqueeze(0)
        point_vec = point_vec / torch.norm(point_vec, dim=-1, keepdim=True)
        point_vecs_per_frame.append(point_vec)
    point_vecs_per_frame = torch.stack(point_vecs_per_frame, dim=0)
    
    point_vecs = point_map[::4] - cam_origins
    point_vecs = point_vecs / torch.norm(point_vecs, dim=-1, keepdim=True)
    
    cam_rays = torch.rot90(cam_rays, k=-1, dims=[1, 2])
    
    attn_maps = torch.cat((point_vecs, cam_rays), dim=2)
    attn_masks = torch.cat((torch.ones_like(point_vecs), torch.zeros_like(cam_rays)), dim=2)
    
    return attn_maps, attn_masks, cam_rays, point_vecs_per_frame


def generate_video_core(
    prompt: str,
    exo_video_path: str,
    ego_prior_video_path: str,
    output_path: str,
    seed: int,
    use_gga: bool,
    cos_sim_scaling_factor: float,
    meta: dict = None,
    is_in_the_wild: bool = False,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
):
    """核心视频生成函数"""
    from core.inference.wan import generate_video
    
    global PIPE
    
    if PIPE is None:
        raise ValueError("模型未加载，请先加载模型")
    
    set_seed(seed)
    
    # 计算 GGA 注意力
    if use_gga and meta is not None:
        attn_maps, attn_masks, cam_rays, point_vecs_per_frame = compute_gga_attention(meta, is_in_the_wild)
    else:
        attn_maps = None
        attn_masks = None
        cam_rays = None
        point_vecs_per_frame = None
    
    # 生成视频
    video = generate_video(
        prompt=prompt,
        exo_video_path=exo_video_path,
        ego_prior_video_path=ego_prior_video_path,
        output_path=output_path,
        num_frames=49,
        width=784 + 448,
        height=448,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        fps=30,
        num_videos_per_prompt=1,
        seed=seed,
        attention_GGA=attn_maps.unsqueeze(0) if attn_maps is not None else None,
        attention_mask_GGA=attn_masks.unsqueeze(0) if attn_masks is not None else None,
        point_vecs_per_frame=point_vecs_per_frame,
        cam_rays=cam_rays,
        do_kv_cache=False,
        cos_sim_scaling_factor=cos_sim_scaling_factor,
        pipe=PIPE,
    )
    
    return output_path


def run_in_the_wild_inference(
    example_idx: str,
    seed: int,
    use_gga: bool,
    cos_sim_scaling_factor: float,
    num_inference_steps: int,
    guidance_scale: float,
    progress=gr.Progress()
):
    """In-the-wild 推理"""
    global MODEL_LOADED
    
    if not MODEL_LOADED:
        return None, "❌ 模型未加载，请检查启动日志！"
    
    try:
        # 解析示例索引
        idx = int(example_idx.split(":")[0])
        
        # 加载元数据
        meta_data = load_meta_data(ITW_META_PATH)
        meta = meta_data[idx]
        
        prompt = meta['prompt']
        exo_video_path = meta['exo_path']
        ego_prior_video_path = meta['ego_prior_path']
        take_name = exo_video_path.split('/')[-2]
        
        progress(0.1, desc="准备数据...")
        
        # 创建输出目录
        output_dir = "./results/gradio_outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{take_name}_itw_{seed}.mp4")
        
        progress(0.2, desc="开始生成视频...")
        
        # 生成视频
        result_path = generate_video_core(
            prompt=prompt,
            exo_video_path=exo_video_path,
            ego_prior_video_path=ego_prior_video_path,
            output_path=output_path,
            seed=seed,
            use_gga=use_gga,
            cos_sim_scaling_factor=cos_sim_scaling_factor,
            meta=meta,
            is_in_the_wild=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        progress(1.0, desc="完成！")
        
        return result_path, f"✅ 视频生成成功！保存至: {result_path}"
    
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def run_ego4d_inference(
    example_idx: str,
    seed: int,
    use_gga: bool,
    cos_sim_scaling_factor: float,
    num_inference_steps: int,
    guidance_scale: float,
    progress=gr.Progress()
):
    """Ego-Exo4D 推理"""
    global MODEL_LOADED
    
    if not MODEL_LOADED:
        return None, "❌ 模型未加载，请检查启动日志！"
    
    try:
        # 解析示例索引
        idx = int(example_idx.split(":")[0])
        
        # 加载元数据
        meta_data = load_meta_data(EGO4D_META_PATH)
        meta = meta_data[idx]
        
        prompt = meta['prompt']
        exo_video_path = meta['exo_path']
        ego_prior_video_path = meta['ego_prior_path']
        take_name = exo_video_path.split('/')[-2]
        
        progress(0.1, desc="准备数据...")
        
        # 创建输出目录
        output_dir = "./results/gradio_outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{take_name}_ego4d_{seed}.mp4")
        
        progress(0.2, desc="开始生成视频...")
        
        # 生成视频
        result_path = generate_video_core(
            prompt=prompt,
            exo_video_path=exo_video_path,
            ego_prior_video_path=ego_prior_video_path,
            output_path=output_path,
            seed=seed,
            use_gga=use_gga,
            cos_sim_scaling_factor=cos_sim_scaling_factor,
            meta=meta,
            is_in_the_wild=False,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        progress(1.0, desc="完成！")
        
        return result_path, f"✅ 视频生成成功！保存至: {result_path}"
    
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def preview_example_itw(example_idx: str):
    """预览 In-the-wild 示例"""
    if not example_idx:
        return None, None, ""
    
    try:
        idx = int(example_idx.split(":")[0])
        meta_data = load_meta_data(ITW_META_PATH)
        meta = meta_data[idx]
        
        exo_video_path = meta['exo_path']
        ego_prior_video_path = meta['ego_prior_path']
        prompt = meta['prompt']
        
        return exo_video_path, ego_prior_video_path, prompt
    except Exception as e:
        return None, None, f"加载失败: {str(e)}"


def preview_example_ego4d(example_idx: str):
    """预览 Ego-Exo4D 示例"""
    if not example_idx:
        return None, None, ""
    
    try:
        idx = int(example_idx.split(":")[0])
        meta_data = load_meta_data(EGO4D_META_PATH)
        meta = meta_data[idx]
        
        exo_video_path = meta['exo_path']
        ego_prior_video_path = meta['ego_prior_path']
        prompt = meta['prompt']
        
        return exo_video_path, ego_prior_video_path, prompt
    except Exception as e:
        return None, None, f"加载失败: {str(e)}"


def create_ui():
    """创建 Gradio 界面"""
    
    # 获取示例选项
    itw_choices = get_example_choices(ITW_META_PATH)
    ego4d_choices = get_example_choices(EGO4D_META_PATH)
    
    with gr.Blocks(title="EgoX - 第一人称视角视频生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 EgoX - 第一人称视角视频生成
        
        基于 Wan2.1 模型，将第三人称（外部视角）视频转换为第一人称（自我视角）视频。
        
        **使用说明：**
        1. 选择一个示例数据
        2. 调整参数（可选）
        3. 点击"生成视频"开始生成
        
        ✅ **模型已自动加载完成**
        """)
        
        # 主要功能区域 - 使用 Tabs
        with gr.Tabs():
            # In-the-wild Tab
            with gr.TabItem("🎭 In-the-Wild 示例"):
                gr.Markdown("""
                **In-the-Wild 数据集** 包含来自电影、动画等多样化场景的视频示例。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📋 选择示例")
                        itw_example_dropdown = gr.Dropdown(
                            choices=itw_choices,
                            label="选择示例",
                            value=itw_choices[0] if itw_choices else None,
                            interactive=True
                        )
                        
                        gr.Markdown("#### 🎛️ 生成参数")
                        itw_seed = gr.Number(label="随机种子", value=846514, precision=0)
                        itw_use_gga = gr.Checkbox(label="使用 GGA (Geometry-Guided Attention)", value=True)
                        itw_cos_sim = gr.Slider(
                            label="余弦相似度缩放因子",
                            minimum=0.1,
                            maximum=10.0,
                            value=3.0,
                            step=0.1
                        )
                        itw_steps = gr.Slider(
                            label="推理步数",
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5
                        )
                        itw_guidance = gr.Slider(
                            label="引导强度 (Guidance Scale)",
                            minimum=1.0,
                            maximum=15.0,
                            value=5.0,
                            step=0.5
                        )
                        
                        itw_generate_btn = gr.Button("🎬 生成视频", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📹 输入视频预览")
                        with gr.Row():
                            itw_exo_video = gr.Video(label="外部视角 (Exo View)", interactive=False)
                            itw_ego_prior_video = gr.Video(label="先验自我视角 (Ego Prior)", interactive=False)
                        
                        gr.Markdown("#### 📝 提示词")
                        itw_prompt_display = gr.Textbox(
                            label="Prompt",
                            lines=5,
                            interactive=False
                        )
                
                gr.Markdown("#### 🎞️ 生成结果")
                itw_output_video = gr.Video(label="生成的视频")
                itw_status = gr.Textbox(label="状态信息", interactive=False)
                
                # 事件绑定
                itw_example_dropdown.change(
                    fn=preview_example_itw,
                    inputs=[itw_example_dropdown],
                    outputs=[itw_exo_video, itw_ego_prior_video, itw_prompt_display]
                )
                
                itw_generate_btn.click(
                    fn=run_in_the_wild_inference,
                    inputs=[
                        itw_example_dropdown,
                        itw_seed,
                        itw_use_gga,
                        itw_cos_sim,
                        itw_steps,
                        itw_guidance
                    ],
                    outputs=[itw_output_video, itw_status]
                )
            
            # Ego-Exo4D Tab
            with gr.TabItem("⚽ Ego-Exo4D 示例"):
                gr.Markdown("""
                **Ego-Exo4D 数据集** 包含来自 Ego-Exo4D 数据集的真实活动视频，如足球、篮球、烹饪、舞蹈等。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📋 选择示例")
                        ego4d_example_dropdown = gr.Dropdown(
                            choices=ego4d_choices,
                            label="选择示例",
                            value=ego4d_choices[0] if ego4d_choices else None,
                            interactive=True
                        )
                        
                        gr.Markdown("#### 🎛️ 生成参数")
                        ego4d_seed = gr.Number(label="随机种子", value=42, precision=0)
                        ego4d_use_gga = gr.Checkbox(label="使用 GGA (Geometry-Guided Attention)", value=True)
                        ego4d_cos_sim = gr.Slider(
                            label="余弦相似度缩放因子",
                            minimum=0.1,
                            maximum=10.0,
                            value=3.0,
                            step=0.1
                        )
                        ego4d_steps = gr.Slider(
                            label="推理步数",
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5
                        )
                        ego4d_guidance = gr.Slider(
                            label="引导强度 (Guidance Scale)",
                            minimum=1.0,
                            maximum=15.0,
                            value=5.0,
                            step=0.5
                        )
                        
                        ego4d_generate_btn = gr.Button("🎬 生成视频", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📹 输入视频预览")
                        with gr.Row():
                            ego4d_exo_video = gr.Video(label="外部视角 (Exo View)", interactive=False)
                            ego4d_ego_prior_video = gr.Video(label="先验自我视角 (Ego Prior)", interactive=False)
                        
                        gr.Markdown("#### 📝 提示词")
                        ego4d_prompt_display = gr.Textbox(
                            label="Prompt",
                            lines=5,
                            interactive=False
                        )
                
                gr.Markdown("#### 🎞️ 生成结果")
                ego4d_output_video = gr.Video(label="生成的视频")
                ego4d_status = gr.Textbox(label="状态信息", interactive=False)
                
                # 事件绑定
                ego4d_example_dropdown.change(
                    fn=preview_example_ego4d,
                    inputs=[ego4d_example_dropdown],
                    outputs=[ego4d_exo_video, ego4d_ego_prior_video, ego4d_prompt_display]
                )
                
                ego4d_generate_btn.click(
                    fn=run_ego4d_inference,
                    inputs=[
                        ego4d_example_dropdown,
                        ego4d_seed,
                        ego4d_use_gga,
                        ego4d_cos_sim,
                        ego4d_steps,
                        ego4d_guidance
                    ],
                    outputs=[ego4d_output_video, ego4d_status]
                )
        
        gr.Markdown("""
        ---
        ### 📖 参数说明
        
        | 参数 | 说明 |
        |------|------|
        | **随机种子** | 控制生成的随机性，相同种子会产生相同结果 |
        | **GGA** | Geometry-Guided Attention，利用几何信息引导视角转换 |
        | **余弦相似度缩放因子** | 控制 GGA 注意力的强度 |
        | **推理步数** | 扩散模型的去噪步数，越多质量越好但越慢 |
        | **引导强度** | 控制生成结果与提示词的匹配程度 |
        
        ---
        **注意：** 视频生成需要较长时间（约 5-10 分钟），请耐心等待。
        """)
    
    return demo


if __name__ == "__main__":
    print("="*50)
    print("🚀 EgoX 启动中...")
    print("="*50)
    
    # 启动时加载模型
    print("\n📦 正在加载模型，请稍候...")
    result = load_model(DEFAULT_MODEL_PATH, DEFAULT_LORA_PATH)
    print(result)
    
    if not MODEL_LOADED:
        print("\n⚠️ 警告：模型加载失败，部分功能可能不可用")
    else:
        print("\n✅ 模型加载完成，正在启动 Web 界面...")
    
    print("="*50)
    
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
