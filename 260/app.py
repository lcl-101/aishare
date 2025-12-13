# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Gradio Web UI for Wan-Move: Motion-Controllable Image-to-Video Generation
"""
import os
import sys
import logging
import tempfile
import json
from datetime import datetime

import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video
from wan.modules.trajectory import draw_tracks_on_video
import torchvision.transforms.functional as TF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# Global model instance
wan_move_model = None
CHECKPOINT_DIR = "./checkpoints/Wan-Move-14B-480P"


def load_model():
    """Load the Wan-Move model."""
    global wan_move_model
    
    if wan_move_model is not None:
        return wan_move_model
    
    logging.info("Loading Wan-Move model...")
    cfg = WAN_CONFIGS['wan-move-i2v']
    
    wan_move_model = wan.WanMove(
        config=cfg,
        checkpoint_dir=CHECKPOINT_DIR,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    
    logging.info("Wan-Move model loaded successfully!")
    return wan_move_model


def interpolate_trajectory(points, num_frames):
    """
    将用户绘制的轨迹点插值到指定帧数。
    
    Args:
        points: 用户绘制的点列表 [(x1, y1), (x2, y2), ...]
        num_frames: 目标帧数
    
    Returns:
        interpolated: 插值后的轨迹 [num_frames, 2]
    """
    if len(points) < 2:
        # 如果只有一个点，复制到所有帧
        return np.array([points[0]] * num_frames)
    
    points = np.array(points)
    
    # 计算累积距离作为参数
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    # 归一化距离到 [0, 1]
    if distances[-1] > 0:
        distances = distances / distances[-1]
    else:
        distances = np.linspace(0, 1, len(points))
    
    # 创建插值函数
    interp_x = interp1d(distances, points[:, 0], kind='linear', fill_value='extrapolate')
    interp_y = interp1d(distances, points[:, 1], kind='linear', fill_value='extrapolate')
    
    # 在均匀分布的参数上插值
    t = np.linspace(0, 1, num_frames)
    interpolated = np.stack([interp_x(t), interp_y(t)], axis=1)
    
    return interpolated


def create_trajectory_from_drawing(drawing_data, image_size, num_frames=81):
    """
    从绘图数据创建轨迹和可见性数组。
    
    Args:
        drawing_data: Gradio ImageEditor 返回的数据
        image_size: 原始图片尺寸 (width, height)
        num_frames: 视频帧数
    
    Returns:
        track: 轨迹数组 [1, num_frames, num_points, 2]
        visibility: 可见性数组 [1, num_frames, num_points]
    """
    if drawing_data is None:
        return None, None
    
    # 从 composite 图层提取轨迹
    # Gradio ImageEditor 返回的数据结构
    if isinstance(drawing_data, dict):
        # 新版 Gradio 格式
        if 'composite' in drawing_data:
            composite = drawing_data['composite']
        else:
            composite = drawing_data.get('image', None)
        
        layers = drawing_data.get('layers', [])
    else:
        return None, None
    
    all_trajectories = []
    
    # 从 layers 中提取绘制的路径
    for layer in layers:
        if isinstance(layer, np.ndarray):
            # 分析图层找到绘制的轨迹
            points = extract_points_from_layer(layer)
            if points and len(points) >= 2:
                all_trajectories.append(points)
    
    if not all_trajectories:
        return None, None
    
    # 为每条轨迹创建插值
    num_points = len(all_trajectories)
    track = np.zeros((1, num_frames, num_points, 2), dtype=np.float32)
    visibility = np.ones((1, num_frames, num_points), dtype=bool)
    
    for i, points in enumerate(all_trajectories):
        interpolated = interpolate_trajectory(points, num_frames)
        track[0, :, i, :] = interpolated
    
    return track, visibility


def extract_points_from_layer(layer):
    """
    从绘图图层中提取轨迹点。
    通过分析非透明像素来找到绘制的路径。
    """
    if layer is None or len(layer.shape) < 3:
        return []
    
    # 获取 alpha 通道或非零像素
    if layer.shape[2] == 4:
        alpha = layer[:, :, 3]
    else:
        # RGB 图层，找非黑色像素
        alpha = np.any(layer > 10, axis=2).astype(np.uint8) * 255
    
    # 找到所有非透明像素的坐标
    y_coords, x_coords = np.where(alpha > 128)
    
    if len(x_coords) == 0:
        return []
    
    # 使用连通性分析来排序点，形成路径
    points = list(zip(x_coords, y_coords))
    
    if len(points) < 2:
        return points
    
    # 简单的排序：按照从左到右或从上到下的顺序
    # 更好的方法是使用最近邻算法
    sorted_points = sort_points_by_path(points)
    
    # 下采样以减少点数
    if len(sorted_points) > 100:
        indices = np.linspace(0, len(sorted_points)-1, 100, dtype=int)
        sorted_points = [sorted_points[i] for i in indices]
    
    return sorted_points


def sort_points_by_path(points):
    """
    使用最近邻算法将散乱的点排序成路径。
    """
    if len(points) <= 2:
        return points
    
    points = list(points)
    sorted_points = [points.pop(0)]
    
    while points:
        last = sorted_points[-1]
        # 找最近的点
        min_dist = float('inf')
        min_idx = 0
        for i, p in enumerate(points):
            dist = (p[0] - last[0])**2 + (p[1] - last[1])**2
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        sorted_points.append(points.pop(min_idx))
    
    return sorted_points


def process_drawing_input(image_with_drawing, num_frames=81):
    """
    处理带绘图的图片输入，返回轨迹数据。
    """
    if image_with_drawing is None:
        return None, None, None
    
    if isinstance(image_with_drawing, dict):
        # ImageEditor 格式
        background = image_with_drawing.get('background', None)
        layers = image_with_drawing.get('layers', [])
        composite = image_with_drawing.get('composite', None)
        
        if background is not None:
            if isinstance(background, np.ndarray):
                original_image = Image.fromarray(background)
            else:
                original_image = background
        elif composite is not None:
            if isinstance(composite, np.ndarray):
                original_image = Image.fromarray(composite)
            else:
                original_image = composite
        else:
            return None, None, None
        
        image_size = original_image.size if hasattr(original_image, 'size') else (original_image.shape[1], original_image.shape[0])
        
        # 提取轨迹
        track, visibility = create_trajectory_from_drawing(
            image_with_drawing, image_size, num_frames
        )
        
        return original_image, track, visibility
    
    return None, None, None


def generate_video_from_drawing(
    image_with_drawing,
    prompt,
    size,
    frame_num,
    sample_steps,
    sample_shift,
    guide_scale,
    seed,
    offload_model,
    vis_track,
    progress=gr.Progress(track_tqdm=True)
):
    """Generate video from image with drawn trajectory."""
    
    if image_with_drawing is None:
        gr.Warning("请上传图片并绘制轨迹!")
        return None, None
    
    if not prompt.strip():
        gr.Warning("请输入提示词!")
        return None, None
    
    try:
        # 处理绘图输入
        img, track, visibility = process_drawing_input(image_with_drawing, frame_num)
        
        if img is None:
            gr.Warning("无法读取图片!")
            return None, None
        
        if track is None or visibility is None:
            gr.Warning("请在图片上绘制轨迹! 使用画笔工具画出物体运动的路径。")
            return None, None
        
        logging.info(f"Track shape from drawing: {track.shape}")
        logging.info(f"Visibility shape: {visibility.shape}")
        
        # Load model
        model = load_model()
        cfg = WAN_CONFIGS['wan-move-i2v']
        
        # Ensure img is PIL Image with RGB (3 channels, no alpha)
        if isinstance(img, np.ndarray):
            # Handle RGBA -> RGB conversion
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            img = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            img = Image.fromarray(np.array(img)).convert("RGB")
        
        # 生成轨迹预览
        preview_img = visualize_trajectory_on_image(img, track, visibility)
        
        video_path = _generate_video_core(
            model, cfg, img, track, visibility,
            prompt, size, frame_num, sample_steps, sample_shift,
            guide_scale, seed, offload_model, vis_track
        )
        
        return video_path, preview_img
        
    except Exception as e:
        logging.exception(f"Error generating video: {e}")
        gr.Error(f"生成视频时出错: {str(e)}")
        return None, None


def visualize_trajectory_on_image(img, track, visibility):
    """在图片上可视化轨迹。"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # track shape: [1, frames, num_points, 2]
    num_points = track.shape[2]
    num_frames = track.shape[1]
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255)
    ]
    
    for p in range(num_points):
        color = colors[p % len(colors)]
        points = []
        for f in range(num_frames):
            if visibility[0, f, p]:
                x, y = track[0, f, p, 0], track[0, f, p, 1]
                points.append((x, y))
        
        # 绘制轨迹线
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        
        # 绘制起点和终点
        if points:
            # 起点 - 圆形
            start = points[0]
            draw.ellipse([start[0]-6, start[1]-6, start[0]+6, start[1]+6], 
                        fill=(0, 255, 0), outline=(255, 255, 255))
            # 终点 - 方形
            end = points[-1]
            draw.rectangle([end[0]-6, end[1]-6, end[0]+6, end[1]+6], 
                          fill=(255, 0, 0), outline=(255, 255, 255))
    
    return img_copy


def _generate_video_core(
    model, cfg, img, track, track_visibility,
    prompt, size, frame_num, sample_steps, sample_shift,
    guide_scale, seed, offload_model, vis_track
):
    """Core video generation logic."""
    
    # Get target size from config
    target_h, target_w = SIZE_CONFIGS[size]
    original_w, original_h = img.size
    
    # Resize image to target size
    if (original_w, original_h) != (target_w, target_h):
        logging.info(f"Resizing image from {original_w}x{original_h} to {target_w}x{target_h}")
        
        # Calculate scale factors
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        
        # Resize image
        img = img.resize((target_w, target_h), Image.LANCZOS)
        
        # Scale track coordinates
        track = track.copy()
        track[:, :, :, 0] = track[:, :, :, 0] * scale_x  # x coordinates
        track[:, :, :, 1] = track[:, :, :, 1] * scale_y  # y coordinates
        
        logging.info(f"Scaled track coordinates by ({scale_x:.3f}, {scale_y:.3f})")
    
    logging.info(f"Input prompt: {prompt}")
    logging.info(f"Track shape: {track.shape}")
    logging.info(f"Track visibility shape: {track_visibility.shape}")
    logging.info(f"Image size: {img.size}")
    
    # Set seed
    if seed < 0:
        seed = torch.randint(0, 2**31, (1,)).item()
    
    logging.info(f"Using seed: {seed}")
    
    # Generate video
    video = model.generate(
        input_prompt=prompt,
        img=img,
        track=track,
        track_visibility=track_visibility,
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver='unipc',
        sampling_steps=sample_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
        eval_bench=True
    )
    
    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    if vis_track:
        # Create track visualization video
        device = torch.device("cuda:0")
        first_frame_repeat = torch.as_tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).repeat(1, frame_num, 1, 1, 1)
        track_video = draw_tracks_on_video(
            first_frame_repeat, 
            torch.from_numpy(track) if isinstance(track, np.ndarray) else track, 
            torch.from_numpy(track_visibility) if isinstance(track_visibility, np.ndarray) else track_visibility
        )
        track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).permute(1, 0, 2, 3).mul(2).sub(1).to(device)
        
        save_file = os.path.join(output_dir, f"wan_move_{timestamp}_with_track.mp4")
        cache_video(
            tensor=torch.stack([track_video, video]),
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
    else:
        save_file = os.path.join(output_dir, f"wan_move_{timestamp}.mp4")
        cache_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
    
    logging.info(f"Video saved to: {save_file}")
    return save_file


# Create Gradio interface
def create_ui():
    with gr.Blocks(
        title="Wan-Move: Motion-Controllable Image-to-Video Generation",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🎬 Wan-Move: Motion-Controllable Image-to-Video Generation
        
        通过轨迹点控制视频中物体的运动方向和路径，将静态图像转化为动态视频。
        
        ### 使用说明
        1. 上传一张图片
        2. 使用画笔在图片上绘制物体运动轨迹（从起点画到终点）
        3. 输入描述视频内容的提示词
        4. 点击"生成视频"
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入")
                
                image_editor = gr.ImageEditor(
                    label="上传图片并绘制轨迹",
                    type="numpy",
                    height=400,
                    brush=gr.Brush(colors=["#FF0000", "#00FF00", "#0000FF"], default_size=5),
                    eraser=gr.Eraser(default_size=10),
                    layers=True,
                )
                
                gr.Markdown("""
                💡 **绘制提示**:
                - 用画笔画出物体运动的路径
                - 从起点画到终点，线条方向就是运动方向
                - 可以画多条轨迹控制多个物体
                - 不同颜色代表不同轨迹
                """)
                
                prompt_input = gr.Textbox(
                    label="提示词 (Prompt)",
                    placeholder="描述视频内容...",
                    lines=3
                )
                
                with gr.Accordion("⚙️ 高级设置", open=False):
                    size_dropdown = gr.Dropdown(
                        label="视频尺寸",
                        choices=["480*832", "832*480"],
                        value="480*832"
                    )
                    
                    frame_num_slider = gr.Slider(
                        label="帧数",
                        minimum=17, maximum=81, step=4, value=81,
                        info="帧数应为 4n+1 的形式"
                    )
                    
                    sample_steps_slider = gr.Slider(
                        label="采样步数",
                        minimum=10, maximum=50, step=1, value=40
                    )
                    
                    sample_shift_slider = gr.Slider(
                        label="采样偏移",
                        minimum=1.0, maximum=10.0, step=0.5, value=3.0
                    )
                    
                    guide_scale_slider = gr.Slider(
                        label="引导强度",
                        minimum=1.0, maximum=15.0, step=0.5, value=5.0
                    )
                    
                    seed_input = gr.Number(label="随机种子", value=-1, precision=0, info="-1 表示随机种子")
                    offload_checkbox = gr.Checkbox(label="模型卸载", value=True, info="启用以减少 GPU 显存使用")
                    vis_track_checkbox = gr.Checkbox(label="可视化轨迹", value=False, info="在输出中显示轨迹可视化")
                
                generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输出")
                
                trajectory_preview = gr.Image(
                    label="轨迹预览",
                    height=200
                )
                
                video_output = gr.Video(
                    label="生成的视频",
                    height=350
                )
        
        generate_btn.click(
            fn=generate_video_from_drawing,
            inputs=[
                image_editor, prompt_input, size_dropdown, frame_num_slider,
                sample_steps_slider, sample_shift_slider, guide_scale_slider,
                seed_input, offload_checkbox, vis_track_checkbox
            ],
            outputs=[video_output, trajectory_preview]
        )
        
        gr.Markdown("""
        ---
        **Wan-Move** - 由阿里巴巴 Wan Team 开发
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
