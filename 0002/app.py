"""
VideoMaMa Gradio Demo
交互式视频抠图与 SAM2 掩码跟踪
"""

import sys
sys.path.append("./")
sys.path.append("./demo")

import os
import json
import time
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from demo.tools.painter import mask_painter, point_painter
from pipeline_svd_mask import VideoInferencePipeline
from sam2.build_sam import build_sam2_video_predictor


class SAM2VideoTracker:
    """SAM2 视频跟踪器包装类"""
    
    def __init__(self, checkpoint_path, config_file, device="cuda"):
        """
        初始化 SAM2 视频跟踪器
        
        Args:
            checkpoint_path: SAM2 检查点路径
            config_file: SAM2 配置文件路径
            device: 运行设备
        """
        self.device = device
        self.predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=device
        )
        print(f"SAM2 视频跟踪器已在 {device} 上初始化")
    
    def track_video(self, frames, points, labels):
        """
        使用 SAM2 跟踪视频中的对象
        
        Args:
            frames: numpy 数组列表, [(H,W,3)]*n, uint8 RGB 帧
            points: 提示点的 [x, y] 坐标列表
            labels: 标签列表 (1 为正向, 0 为负向)
            
        Returns:
            masks: numpy 数组列表, [(H,W)]*n, uint8 二值掩码
        """
        # 创建临时目录存放帧
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        try:
            # 保存帧到临时目录
            print(f"正在保存 {len(frames)} 帧到临时目录...")
            for i, frame in enumerate(frames):
                frame_path = frames_dir / f"{i:05d}.jpg"
                Image.fromarray(frame).save(frame_path, quality=95)
            
            # 初始化 SAM2 视频预测器
            print("正在初始化 SAM2 推理状态...")
            inference_state = self.predictor.init_state(video_path=str(frames_dir))
            
            # 在第一帧上添加提示
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            
            print(f"在第一帧上添加 {len(points)} 个点提示...")
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_array,
                labels=labels_array,
            )
            
            # 在视频中传播
            print("正在视频中传播掩码...")
            masks = []
            for frame_idx, object_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
                obj_ids_list = object_ids.tolist() if hasattr(object_ids, 'tolist') else object_ids
                
                if 1 in obj_ids_list:
                    mask_idx = obj_ids_list.index(1)
                    mask = (mask_logits[mask_idx] > 0.0).cpu().numpy()
                    mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                    masks.append(mask_uint8)
                else:
                    h, w = frames[0].shape[:2]
                    masks.append(np.zeros((h, w), dtype=np.uint8))
            
            print(f"已生成 {len(masks)} 个掩码")
            return masks
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def get_first_frame_mask(self, frame, points, labels):
        """
        仅获取第一帧的掩码（用于预览）
        
        Args:
            frame: np.ndarray, (H, W, 3), uint8 RGB 帧
            points: [x, y] 坐标列表
            labels: 标签列表 (1 为正向, 0 为负向)
            
        Returns:
            mask: np.ndarray, (H, W), uint8 二值掩码
        """
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        try:
            frame_path = frames_dir / "00000.jpg"
            Image.fromarray(frame).save(frame_path, quality=95)
            
            inference_state = self.predictor.init_state(video_path=str(frames_dir))
            
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_array,
                labels=labels_array,
            )
            
            if len(out_mask_logits) > 0:
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                return mask_uint8
            else:
                return np.zeros(frame.shape[:2], dtype=np.uint8)
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_sam2_tracker(device="cuda"):
    """
    加载 SAM2 视频跟踪器
    
    Args:
        device: 运行设备
        
    Returns:
        SAM2VideoTracker 实例
    """
    checkpoint_path = "checkpoints/sam2/sam2.1_hiera_large.pt"
    # 使用 SAM2 包内的配置路径格式
    config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    print(f"正在从 {checkpoint_path} 加载 SAM2...")
    tracker = SAM2VideoTracker(checkpoint_path, config_file, device)
    
    return tracker


def load_videomama_pipeline(device="cuda"):
    """
    加载 VideoMaMa 管道并使用预训练权重
    
    Args:
        device: 运行设备
        
    Returns:
        VideoInferencePipeline 实例
    """
    # 使用本地 checkpoints 目录
    base_model_path = "checkpoints/stable-video-diffusion-img2vid-xt"
    unet_checkpoint_path = "checkpoints/VideoMaMa"
    
    print(f"正在从 {unet_checkpoint_path} 加载 VideoMaMa 管道...")
    
    pipeline = VideoInferencePipeline(
        base_model_path=base_model_path,
        unet_checkpoint_path=unet_checkpoint_path,
        weight_dtype=torch.float16,
        device=device
    )
    
    print("VideoMaMa 管道加载成功！")
    
    return pipeline


def videomama(pipeline, frames_np, mask_frames_np):
    """
    使用掩码条件运行 VideoMaMa 推理
    
    Args:
        pipeline: VideoInferencePipeline 实例
        frames_np: numpy 数组列表, [(H,W,3)]*n, uint8 RGB 帧
        mask_frames_np: numpy 数组列表, [(H,W)]*n, uint8 灰度掩码
        
    Returns:
        output_frames: numpy 数组列表, [(H,W,3)]*n, uint8 RGB 输出
    """
    # 将 numpy 数组转换为 PIL 图像
    frames_pil = [Image.fromarray(f) for f in frames_np]
    mask_frames_pil = [Image.fromarray(m, mode='L') for m in mask_frames_np]
    
    # 调整到模型输入大小
    target_width, target_height = 1024, 576
    frames_resized = [f.resize((target_width, target_height), Image.Resampling.BILINEAR) 
                     for f in frames_pil]
    masks_resized = [m.resize((target_width, target_height), Image.Resampling.BILINEAR) 
                    for m in mask_frames_pil]
    
    # 运行推理
    print(f"在 {len(frames_resized)} 帧上运行 VideoMaMa 推理...")
    output_frames_pil = pipeline.run(
        cond_frames=frames_resized,
        mask_frames=masks_resized,
        seed=42,
        mask_cond_mode="vae"
    )
    
    # 调整回原始分辨率
    original_size = frames_pil[0].size
    output_frames_resized = [f.resize(original_size, Image.Resampling.BILINEAR) 
                            for f in output_frames_pil]
    
    # 转换回 numpy 数组
    output_frames_np = [np.array(f) for f in output_frames_resized]
    
    return output_frames_np

import warnings
warnings.filterwarnings("ignore")

# 全局模型
sam2_tracker = None
videomama_pipeline = None

# 常量
MASK_COLOR = 3
MASK_ALPHA = 0.7
CONTOUR_COLOR = 1
CONTOUR_WIDTH = 5
POINT_COLOR_POS = 8   # 正向点 - 橙色
POINT_COLOR_NEG = 1   # 负向点 - 红色
POINT_ALPHA = 0.9
POINT_RADIUS = 15

def initialize_models():
    """初始化 SAM2 和 VideoMaMa 模型"""
    global sam2_tracker, videomama_pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载 SAM2
    sam2_tracker = load_sam2_tracker(device=device)
    
    # 加载 VideoMaMa
    videomama_pipeline = load_videomama_pipeline(device=device)
    
    print("所有模型初始化成功！")


def extract_frames_from_video(video_path, max_frames=24):
    """
    从视频文件中提取帧
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大提取帧数（默认：24）
        
    Returns:
        frames: numpy 数组列表 (H,W,3), uint8 RGB
        adjusted_fps: 调整后的 FPS，用于输出视频保持正常播放速度
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 首先读取所有帧
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 将 BGR 转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
    
    cap.release()
    
    # 如果视频帧数超过 max_frames，则随机采样
    if len(all_frames) > max_frames:
        print(f"视频有 {len(all_frames)} 帧，随机采样 {max_frames} 帧...")
        # 排序索引以保持时间顺序
        sampled_indices = sorted(np.random.choice(len(all_frames), max_frames, replace=False))
        frames = [all_frames[i] for i in sampled_indices]
        print(f"采样帧索引: {sampled_indices}")
        
        # 调整 FPS 以保持正常播放速度
        # 如果从 M 总帧中采样了 N 帧，则按比例调整 FPS
        adjusted_fps = original_fps * (len(frames) / len(all_frames))
    else:
        frames = all_frames
        adjusted_fps = original_fps
        print(f"视频有 {len(frames)} 帧 (≤ {max_frames})，使用全部帧")
    
    print(f"使用视频的 {len(frames)} 帧 (原始 FPS: {original_fps:.2f}，调整后 FPS: {adjusted_fps:.2f})")
    
    return frames, adjusted_fps


def get_prompt(click_state, click_input):
    """
    将点击输入转换为提示格式
    
    Args:
        click_state: [[points], [labels]]
        click_input: JSON 字符串 "[[x, y, label]]"
        
    Returns:
        更新后的 click_state
    """
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    
    for input_item in inputs:
        points.append(input_item[:2])
        labels.append(input_item[2])
    
    click_state[0] = points
    click_state[1] = labels
    
    return click_state


def get_video_info(video_path):
    """
    获取视频信息（帧数、FPS等）
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        total_frames: 总帧数
        fps: 帧率
        duration: 时长（秒）
    """
    if video_path is None:
        return 0, 0, 0
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    return total_frames, fps, duration


def on_video_upload(video_input):
    """
    当用户上传视频时，自动获取视频信息并更新滑块
    """
    if video_input is None:
        return gr.update(maximum=100, value=24, info="从视频中均匀采样的帧数。推荐 24-50 帧，更多帧需要更多显存。"), \
               ""
    
    total_frames, fps, duration = get_video_info(video_input)
    
    if total_frames == 0:
        return gr.update(maximum=100, value=24, info="从视频中均匀采样的帧数。推荐 24-50 帧，更多帧需要更多显存。"), \
               ""
    
    # 计算推荐帧数：最多50帧，或者视频本身的帧数（如果少于50）
    recommended_frames = min(50, total_frames)
    
    # 更新滑块的最大值为视频的总帧数（但不超过200帧以防显存溢出）
    max_frames = min(total_frames, 200)
    
    # 生成视频信息文字
    video_info = f"📹 视频信息：共 {total_frames} 帧 | {fps:.1f} FPS | 时长 {duration:.1f} 秒"
    
    slider_info = f"从视频中均匀采样的帧数。推荐 {recommended_frames} 帧（显存充足可增加，建议不超过100帧）"
    
    return gr.update(maximum=max_frames, value=recommended_frames, info=slider_info), \
           video_info


def load_video(video_input, video_state, num_frames):
    """
    加载视频并提取第一帧用于生成掩码
    """
    # 清理旧的输出文件（如果存在）
    if video_state is not None and "output_paths" in video_state:
        cleanup_old_videos(video_state["output_paths"])
    
    if video_input is None:
        return video_state, None, \
               gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=False), gr.update(visible=False)
    
    # 使用用户指定的帧数提取帧
    frames, fps = extract_frames_from_video(video_input, max_frames=num_frames)
    
    if len(frames) == 0:
        return video_state, None, \
               gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=False), gr.update(visible=False)
    
    # 初始化视频状态
    video_state = {
        "frames": frames,
        "fps": fps,
        "first_frame_mask": None,
        "masks": None,
    }
    
    first_frame_pil = Image.fromarray(frames[0])
    
    return video_state, first_frame_pil, \
           gr.update(visible=True), gr.update(visible=True), \
           gr.update(visible=True), gr.update(visible=False)


def sam_refine(video_state, point_prompt, click_state, evt: gr.SelectData):
    """
    添加点击并更新第一帧上的掩码
    
    Args:
        video_state: 包含视频数据的字典
        point_prompt: "正向点" 或 "负向点"
        click_state: [[points], [labels]]
        evt: Gradio SelectData 事件，包含点击坐标
    """
    if video_state is None or "frames" not in video_state:
        return None, video_state, click_state
    
    # 添加新点击
    x, y = evt.index[0], evt.index[1]
    label = 1 if point_prompt == "正向点" else 0
    
    click_state[0].append([x, y])
    click_state[1].append(label)
    
    print(f"添加 {point_prompt} 点击位置 ({x}, {y})。总点击数: {len(click_state[0])}")
    
    # 使用 SAM2 生成掩码
    first_frame = video_state["frames"][0]
    mask = sam2_tracker.get_first_frame_mask(
        frame=first_frame,
        points=click_state[0],
        labels=click_state[1]
    )
    
    # 将掩码存储在视频状态中
    video_state["first_frame_mask"] = mask
    
    # 可视化掩码和点
    painted_image = mask_painter(
        first_frame.copy(),
        mask,
        MASK_COLOR,
        MASK_ALPHA,
        CONTOUR_COLOR,
        CONTOUR_WIDTH
    )
    
    # 绘制正向点
    positive_points = np.array([click_state[0][i] for i in range(len(click_state[0])) 
                               if click_state[1][i] == 1])
    if len(positive_points) > 0:
        painted_image = point_painter(
            painted_image,
            positive_points,
            POINT_COLOR_POS,
            POINT_ALPHA,
            POINT_RADIUS,
            CONTOUR_COLOR,
            CONTOUR_WIDTH
        )
    
    # 绘制负向点
    negative_points = np.array([click_state[0][i] for i in range(len(click_state[0])) 
                               if click_state[1][i] == 0])
    if len(negative_points) > 0:
        painted_image = point_painter(
            painted_image,
            negative_points,
            POINT_COLOR_NEG,
            POINT_ALPHA,
            POINT_RADIUS,
            CONTOUR_COLOR,
            CONTOUR_WIDTH
        )
    
    painted_pil = Image.fromarray(painted_image)
    
    return painted_pil, video_state, click_state


def clear_clicks(video_state, click_state):
    """清除所有点击并重置为原始第一帧"""
    click_state = [[], []]
    
    if video_state is not None and "frames" in video_state:
        first_frame = video_state["frames"][0]
        video_state["first_frame_mask"] = None
        return Image.fromarray(first_frame), video_state, click_state
    
    return None, video_state, click_state


def propagate_masks(video_state, click_state):
    """
    使用 SAM2 在整个视频中传播第一帧掩码
    """
    if video_state is None or "frames" not in video_state:
        return video_state, "未加载视频", gr.update(visible=False)
    
    if len(click_state[0]) == 0:
        return video_state, "⚠️ 请先添加至少一个点", gr.update(visible=False)
    
    frames = video_state["frames"]
    
    # 在视频中跟踪
    print(f"在 {len(frames)} 帧中跟踪对象...")
    masks = sam2_tracker.track_video(
        frames=frames,
        points=click_state[0],
        labels=click_state[1]
    )
    
    video_state["masks"] = masks
    
    status_msg = f"✓ 已生成 {len(masks)} 个掩码。准备运行 VideoMaMa！"
    
    return video_state, status_msg, gr.update(visible=True)


def run_videomama_with_sam2(video_state, click_state):
    """
    一起运行 SAM2 传播和 VideoMaMa 推理
    """
    if video_state is None or "frames" not in video_state:
        return video_state, None, None, None, "⚠️ 未加载视频"
    
    if len(click_state[0]) == 0:
        return video_state, None, None, None, "⚠️ 请先添加至少一个点"
    
    frames = video_state["frames"]
    
    # 步骤 1: 使用 SAM2 在视频中跟踪
    print(f"🎯 使用 SAM2 在 {len(frames)} 帧中跟踪对象...")
    masks = sam2_tracker.track_video(
        frames=frames,
        points=click_state[0],
        labels=click_state[1]
    )
    
    video_state["masks"] = masks
    print(f"✓ 已生成 {len(masks)} 个掩码")
    
    # 步骤 2: 运行 VideoMaMa
    print(f"🎨 在 {len(frames)} 帧上运行 VideoMaMa...")
    output_frames = videomama(videomama_pipeline, frames, masks)
    
    # 保存输出视频
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    output_video_path = output_dir / f"output_{timestamp}.mp4"
    mask_video_path = output_dir / f"masks_{timestamp}.mp4"
    greenscreen_path = output_dir / f"greenscreen_{timestamp}.mp4"
    
    # 保存抠图结果
    save_video(output_frames, output_video_path, video_state["fps"])
    
    # 保存掩码视频（用于可视化）
    mask_frames_rgb = [np.stack([m, m, m], axis=-1) for m in masks]
    save_video(mask_frames_rgb, mask_video_path, video_state["fps"])
    
    # 创建绿屏合成：RGB * VideoMaMa_alpha + green * (1 - VideoMaMa_alpha)
    # VideoMaMa output_frames 已包含 alpha 蒙版结果
    greenscreen_frames = []
    for orig_frame, output_frame in zip(frames, output_frames):
        # 从 VideoMaMa 输出中提取 alpha 蒙版
        # VideoMaMa 输出抠图后的前景，我们使用其强度作为 alpha
        gray = cv2.cvtColor(output_frame, cv2.COLOR_RGB2GRAY)
        alpha = np.clip(gray.astype(np.float32) / 255.0, 0, 1)
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
        
        # 创建绿色背景
        green_bg = np.zeros_like(orig_frame)
        green_bg[:, :] = [156, 251, 165]  # 绿屏颜色
        
        # 合成：original_RGB * alpha + green * (1 - alpha)
        composite = (orig_frame.astype(np.float32) * alpha_3ch + 
                    green_bg.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)
        greenscreen_frames.append(composite)
    
    save_video(greenscreen_frames, greenscreen_path, video_state["fps"])
    
    status_msg = f"✓ 完成！已生成 {len(output_frames)} 帧。"
    
    # 存储路径以便稍后清理
    video_state["output_paths"] = [str(output_video_path), str(mask_video_path), str(greenscreen_path)]
    
    return video_state, str(output_video_path), str(mask_video_path), str(greenscreen_path), status_msg


def save_video(frames, output_path, fps):
    """将帧保存为视频文件"""
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        if len(frame.shape) == 2:  # 灰度图
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"已保存视频到 {output_path}")


def cleanup_old_videos(video_paths):
    """删除旧的输出视频以节省存储空间"""
    if video_paths is None:
        return
    
    for path in video_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"已清理: {path}")
        except Exception as e:
            print(f"删除失败 {path}: {e}")


def cleanup_old_outputs(max_age_minutes=30):
    """
    删除超过 max_age_minutes 的输出文件以防止存储溢出
    定期运行以清理废弃文件
    """
    output_dir = Path("outputs")
    if not output_dir.exists():
        return
    
    current_time = time.time()
    max_age_seconds = max_age_minutes * 60
    
    for file_path in output_dir.glob("*.mp4"):
        try:
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                print(f"已清理旧文件: {file_path} (时长: {file_age/60:.1f} 分钟)")
        except Exception as e:
            print(f"清理失败 {file_path}: {e}")


def restart():
    """重置所有状态"""
    return None, [[], []], None, \
           gr.update(visible=False), gr.update(visible=False), \
           gr.update(visible=False), None, None, None, "", \
           gr.update(maximum=100, value=24, info="从视频中均匀采样的帧数。推荐 24-50 帧，更多帧需要更多显存。"), ""


# CSS 样式
custom_css = """
.gradio-container {width: 90% !important; margin: 0 auto;}
.title-text {text-align: center; font-size: 48px; font-weight: bold; 
             background: linear-gradient(to right, #8b5cf6, #10b981); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.description-text {text-align: center; font-size: 18px; margin: 20px 0;}
.youtube-link {
    text-align: center; 
    font-size: 16px; 
    margin: 10px 0 20px 0; 
    padding: 10px;
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    border-radius: 10px;
}
.youtube-link a {
    color: white !important;
    text-decoration: none;
    font-weight: bold;
}
.youtube-link a:hover {
    text-decoration: underline;
}
button {border-radius: 8px !important;}
.green_button {background-color: #10b981 !important; color: white !important;}
.red_button {background-color: #ef4444 !important; color: white !important;}
.run_matting_button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 18px !important;
    padding: 20px !important;
    box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.75) !important;
    border: none !important;
}
.run_matting_button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 50%, #f093fb 100%) !important;
    box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.9) !important;
    transform: translateY(-2px) !important;
}
"""

# 构建 Gradio 界面
with gr.Blocks(css=custom_css, title="VideoMaMa 演示") as demo:
    gr.HTML('<div class="youtube-link">📺 <a href="https://www.youtube.com/@rongyi-ai" target="_blank">AI 技术分享频道</a> - 欢迎订阅！</div>')
    gr.HTML('<div class="title-text">VideoMaMa 交互式演示</div>')
    gr.Markdown(
        '<div class="description-text">🎬 上传视频 → 🖱️ 点击标记对象 → ✅ 生成掩码 → 🎨 运行 VideoMaMa</div>'
    )
    
    # 状态变量
    video_state = gr.State(None)
    click_state = gr.State([[], []])  # [[points], [labels]]
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 步骤 1：上传视频")
            video_input = gr.Video(label="输入视频")
            video_info_text = gr.Markdown("", elem_id="video_info")
            num_frames_slider = gr.Slider(
                minimum=1,
                maximum=100,
                value=24,
                step=1,
                label="要处理的帧数",
                info="从视频中均匀采样的帧数。推荐 24-50 帧，更多帧需要更多显存。"
            )
            load_button = gr.Button("📁 加载视频", variant="primary")
            
            gr.Markdown("### 步骤 2：标记对象")
            point_prompt = gr.Radio(
                choices=["正向点", "负向点"],
                value="正向点",
                label="点击类型",
                info="正向点：对象，负向点：背景",
                visible=False
            )
            clear_button = gr.Button("🗑️ 清除点击", visible=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### 第一帧（点击添加点）")
            first_frame_display = gr.Image(
                label="第一帧",
                type="pil",
                interactive=True
            )
            run_button = gr.Button("🚀 运行抠图", visible=False, elem_classes="run_matting_button", size="lg")
    
    status_text = gr.Textbox(label="状态", value="", interactive=False, visible=False)
    
    gr.Markdown("### 输出结果")
    with gr.Row():
        with gr.Column():
            output_video = gr.Video(label="抠图结果（VideoMaMa 生成）", autoplay=True)
        with gr.Column():
            greenscreen_video = gr.Video(label="绿屏合成", autoplay=True)
        with gr.Column():
            mask_video = gr.Video(label="分割掩码（SAM2 分割）", autoplay=True)
    
    # 事件处理器
    load_button.click(
        fn=load_video,
        inputs=[video_input, video_state, num_frames_slider],
        outputs=[video_state, first_frame_display, 
                point_prompt, clear_button, run_button, status_text]
    )
    
    first_frame_display.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state],
        outputs=[first_frame_display, video_state, click_state]
    )
    
    clear_button.click(
        fn=clear_clicks,
        inputs=[video_state, click_state],
        outputs=[first_frame_display, video_state, click_state]
    )
    
    run_button.click(
        fn=run_videomama_with_sam2,
        inputs=[video_state, click_state],
        outputs=[video_state, output_video, mask_video, greenscreen_video, status_text]
    )
    
    # 视频上传时自动更新帧数信息
    video_input.upload(
        fn=on_video_upload,
        inputs=[video_input],
        outputs=[num_frames_slider, video_info_text]
    )
    
    video_input.change(
        fn=restart,
        inputs=[],
        outputs=[video_state, click_state, first_frame_display,
                point_prompt, clear_button, run_button, 
                output_video, mask_video, greenscreen_video, status_text,
                num_frames_slider, video_info_text]
    )
    
    # 示例
    gr.Markdown("---\n### 📦 示例视频")
    example_dir = Path("assets")
    if example_dir.exists():
        examples = [str(p) for p in sorted(example_dir.glob("*.mp4"))]
        if examples:
            gr.Examples(examples=examples, inputs=[video_input])


if __name__ == "__main__":
    print("=" * 60)
    print("VideoMaMa 交互式演示")
    print("=" * 60)
    
    # 启动时清理旧的输出文件
    cleanup_old_outputs(max_age_minutes=30)
    
    # 初始化模型
    initialize_models()
    
    # 启动演示
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
