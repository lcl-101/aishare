"""
Molmo2 Gradio Web Application

A multi-tab web interface for interacting with Molmo2-8B model supporting:
- General Video QA
- Pointing Video QA
- Tracking Video QA
- Multi-image QA
- Multi-Image Point QA
"""

import os
import re
import torch
import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from transformers import AutoProcessor, AutoModelForImageTextToText

# ============================================================================
# Constants and Configuration
# ============================================================================

MODEL_PATH = "./checkpoints/Molmo2-8B"
EXAMPLES_DIR = "./examples"

# Example URLs to download
EXAMPLE_URLS = {
    "videos": {
        "penguins": "https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4",
        "basketball": "https://storage.googleapis.com/oe-training-public/demo_videos/arena_basketball.mp4",
    },
    "images": {
        "dog": "https://picsum.photos/id/237/536/354",
        "cherry_blossom": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg",
        "boat1": "https://storage.googleapis.com/oe-training-public/demo_images/boat1.jpeg",
        "boat2": "https://storage.googleapis.com/oe-training-public/demo_images/boat2.jpeg",
    }
}

# ============================================================================
# Regex Patterns for Point Extraction
# ============================================================================

COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


# ============================================================================
# Example Download Functions
# ============================================================================

def download_examples():
    """Download example files at startup."""
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXAMPLES_DIR, "videos"), exist_ok=True)
    os.makedirs(os.path.join(EXAMPLES_DIR, "images"), exist_ok=True)
    
    downloaded_files = {"videos": {}, "images": {}}
    
    # Download videos
    for name, url in EXAMPLE_URLS["videos"].items():
        save_path = os.path.join(EXAMPLES_DIR, "videos", f"{name}.mp4")
        if not os.path.exists(save_path):
            print(f"Downloading {name} video...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Saved to {save_path}")
            except Exception as e:
                print(f"  Error downloading {name}: {e}")
                continue
        downloaded_files["videos"][name] = save_path
    
    # Download images
    for name, url in EXAMPLE_URLS["images"].items():
        ext = ".jpg" if "jpg" in url or "jpeg" in url else ".png"
        save_path = os.path.join(EXAMPLES_DIR, "images", f"{name}{ext}")
        if not os.path.exists(save_path):
            print(f"Downloading {name} image...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"  Saved to {save_path}")
            except Exception as e:
                print(f"  Error downloading {name}: {e}")
                continue
        downloaded_files["images"][name] = save_path
    
    return downloaded_files


# ============================================================================
# Point Extraction Functions
# ============================================================================

def _points_from_num_str(text, image_w, image_h, extract_ids=False):
    """Extract points from number string format."""
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        # Our points format assumes coordinates are scaled by 1000
        x, y = float(x) / 1000 * image_w, float(y) / 1000 * image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y


def extract_video_points(text, image_w, image_h, extract_ids=False):
    """
    Extract video pointing coordinates as a flattened list of (t, x, y) triplets from model output text.
    """
    all_points = []
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = float(point_grp.group(1))
            w, h = (image_w, image_h)
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, x, y))
                else:
                    all_points.append((frame_id, x, y))
    return all_points


def extract_multi_image_points(text, image_w, image_h, extract_ids=False):
    """
    从模型输出文本中提取多图指向坐标。
    
    格式示例: <points coords="1 1 098 629 2 162 629...;2 22 142 418...">boats</points>
    - 分号 ; 分隔不同图片
    - 每个图片组的第一个数字是图片索引（1或2）
    - 后续是 "点索引 X坐标 Y坐标" 的重复
    """
    all_points = []
    
    # 判断是否有多个不同尺寸的图片
    if isinstance(image_w, (list, tuple)) and isinstance(image_h, (list, tuple)):
        assert len(image_w) == len(image_h)
        multi_size = True
    else:
        multi_size = False
        image_w = [image_w]
        image_h = [image_h]
    
    # 提取 coords 属性内容
    for coord_match in COORD_REGEX.finditer(text):
        coords_str = coord_match.group(1)
        
        # 按分号分割不同图片的点
        image_groups = coords_str.split(';')
        
        for group in image_groups:
            group = group.strip()
            if not group:
                continue
            
            # 解析数字序列
            numbers = re.findall(r'[0-9]+', group)
            if len(numbers) < 4:  # 至少需要: 图片索引, 点索引, x, y
                continue
            
            # 第一个数字是图片索引
            frame_id = int(numbers[0])
            
            # 获取对应图片的尺寸
            img_idx = frame_id - 1  # 转为 0-indexed
            if img_idx < 0 or img_idx >= len(image_w):
                img_idx = 0  # 回退到第一张图
            w, h = image_w[img_idx], image_h[img_idx]
            
            # 剩余数字每3个一组: 点索引, x, y
            remaining = numbers[1:]
            for i in range(0, len(remaining) - 2, 3):
                try:
                    point_idx = int(remaining[i])
                    x = float(remaining[i + 1]) / 1000 * w
                    y = float(remaining[i + 2]) / 1000 * h
                    
                    if 0 <= x <= w and 0 <= y <= h:
                        if extract_ids:
                            all_points.append((frame_id, point_idx, x, y))
                        else:
                            all_points.append((frame_id, x, y))
                except (ValueError, IndexError):
                    continue
    
    return all_points


def get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Get video dimensions using decord or fallback."""
    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        # Get first frame to determine dimensions
        frame = vr[0].asnumpy()
        height, width = frame.shape[:2]
        return width, height
    except Exception:
        pass
    
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    
    # Default fallback
    return 1920, 1080


def extract_video_frames_with_points(video_path: str, points: list, max_frames: int = 4) -> list[Image.Image]:
    """
    从视频中提取包含标记点的帧，并在帧上绘制点。
    
    Args:
        video_path: 视频文件路径
        points: 点列表，格式为 (timestamp, x, y) 或 (timestamp, idx, x, y)
        max_frames: 最多返回多少帧
    
    Returns:
        标注后的 PIL Image 列表
    """
    if not points:
        return []
    
    # 按时间戳分组点
    from collections import defaultdict
    points_by_time = defaultdict(list)
    for point in points:
        if len(point) >= 3:
            timestamp = float(point[0])
            if len(point) == 3:
                points_by_time[timestamp].append((point[1], point[2]))
            elif len(point) == 4:
                points_by_time[timestamp].append((point[2], point[3]))
    
    # 选择要显示的时间戳（最多 max_frames 个，均匀分布）
    all_timestamps = sorted(points_by_time.keys())
    if len(all_timestamps) <= max_frames:
        selected_timestamps = all_timestamps
    else:
        # 均匀选择
        indices = [int(i * (len(all_timestamps) - 1) / (max_frames - 1)) for i in range(max_frames)]
        selected_timestamps = [all_timestamps[i] for i in indices]
    
    annotated_frames = []
    
    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        
        for timestamp in selected_timestamps:
            # 将时间戳转换为帧索引
            frame_idx = int(timestamp * fps)
            frame_idx = min(frame_idx, len(vr) - 1)
            frame_idx = max(frame_idx, 0)
            
            # 提取帧
            frame = vr[frame_idx].asnumpy()
            frame_image = Image.fromarray(frame)
            
            # 在帧上绘制点
            pts = points_by_time[timestamp]
            if pts:
                frame_image = draw_points_on_image(frame_image, pts)
            
            # 添加时间戳标签
            draw = ImageDraw.Draw(frame_image)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            label = f"t={timestamp:.1f}s"
            # 绘制背景
            draw.rectangle([(5, 5), (150, 35)], fill=(0, 0, 0, 180))
            draw.text((10, 8), label, fill=(255, 255, 255), font=font)
            
            annotated_frames.append(frame_image)
            
    except Exception as e:
        print(f"[ERROR] 提取视频帧失败: {e}")
    
    return annotated_frames


def format_points_output(points: list) -> str:
    """Format points list into readable string."""
    if not points:
        return "No points detected."
    
    lines = []
    for i, point in enumerate(points[:20]):  # Limit to first 20 points
        if len(point) == 3:
            lines.append(f"Point {i+1}: Frame/Image {point[0]:.1f}, X={point[1]:.2f}, Y={point[2]:.2f}")
        elif len(point) == 4:
            lines.append(f"Point {i+1}: Frame/Image {point[0]:.1f}, ID={point[1]}, X={point[2]:.2f}, Y={point[3]:.2f}")
    
    if len(points) > 20:
        lines.append(f"... and {len(points) - 20} more points")
    
    return "\n".join(lines)


# ============================================================================
# Visualization Functions
# ============================================================================

# Color palette for different points/objects
COLOR_PALETTE = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Rose
    (128, 255, 0),    # Lime
    (0, 128, 255),    # Sky Blue
]


def draw_points_on_image(
    image: Image.Image,
    points: list[tuple],
    point_radius: int = None,
    show_labels: bool = True,
    label_offset: int = None,
) -> Image.Image:
    """
    Draw points on an image with labels.
    
    Args:
        image: PIL Image to draw on
        points: List of (x, y) or (idx, x, y) tuples
        point_radius: Radius of the point circle (auto-scaled if None)
        show_labels: Whether to show point number labels
        label_offset: Offset for label text from point
    
    Returns:
        New PIL Image with points drawn
    """
    # Make a copy to avoid modifying original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Auto-scale point size based on image dimensions
    img_size = max(image.width, image.height)
    if point_radius is None:
        point_radius = max(12, img_size // 100)  # 至少12像素，或图片尺寸的1%
    if label_offset is None:
        label_offset = point_radius + 5
    
    # 根据图片大小选择字体大小
    font_size = max(16, img_size // 60)
    
    # Try to get a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    for i, point in enumerate(points):
        # Handle different point formats
        if len(point) == 2:
            x, y = point
        elif len(point) == 3:
            _, x, y = point  # (frame_id, x, y)
        elif len(point) == 4:
            _, _, x, y = point  # (frame_id, idx, x, y)
        else:
            continue
        
        # Get color from palette (cycle if more points than colors)
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        
        # Draw filled circle
        draw.ellipse(
            [(x - point_radius, y - point_radius),
             (x + point_radius, y + point_radius)],
            fill=color,
            outline=(255, 255, 255),
            width=2
        )
        
        # Draw label
        if show_labels:
            label = str(i + 1)
            # Draw text with outline for better visibility
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text(
                    (x + label_offset + dx, y - label_offset + dy),
                    label,
                    fill=(0, 0, 0),
                    font=font
                )
            draw.text(
                (x + label_offset, y - label_offset),
                label,
                fill=color,
                font=font
            )
    
    return annotated


def draw_points_on_multi_images(
    images: list[Image.Image],
    points: list[tuple],
) -> list[Image.Image]:
    """
    Draw points on multiple images based on frame/image index.
    
    Args:
        images: List of PIL Images
        points: List of (frame_id, x, y) or (frame_id, idx, x, y) tuples
                frame_id is 1-indexed
    
    Returns:
        List of annotated PIL Images
    """
    # Group points by image index
    points_per_image = {i: [] for i in range(len(images))}
    
    for point in points:
        if len(point) >= 3:
            frame_id = int(point[0])  # 1-indexed
            image_idx = frame_id - 1   # Convert to 0-indexed
            if 0 <= image_idx < len(images):
                if len(point) == 3:
                    points_per_image[image_idx].append((point[1], point[2]))
                elif len(point) == 4:
                    points_per_image[image_idx].append((point[2], point[3]))
    
    # Debug
    for i, pts in points_per_image.items():
        print(f"[DEBUG] 图片 {i+1} 有 {len(pts)} 个点")
        if pts:
            print(f"[DEBUG]   前3个: {pts[:3]}")
    
    # Draw points on each image
    annotated_images = []
    for i, img in enumerate(images):
        if points_per_image[i]:
            annotated = draw_points_on_image(img, points_per_image[i])  # 自动调整大小
        else:
            annotated = img.copy()
        annotated_images.append(annotated)
    
    return annotated_images


# ============================================================================
# Model Loading
# ============================================================================

# Global model and processor (loaded once)
model = None
processor = None


def load_model():
    """Load the Molmo2 model and processor."""
    global model, processor
    
    if processor is None:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )
        print("Processor loaded.")
    
    if model is None:
        print("Loading model...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Model loaded.")
    
    return model, processor


# ============================================================================
# Inference Functions
# ============================================================================

def general_video_qa(video_path: str, question: str, max_tokens: int = 2048) -> str:
    """
    General Video QA: Answer questions about a video.
    """
    if not video_path:
        return "Please provide a video."
    if not question:
        return "Please provide a question."
    
    model, processor = load_model()
    
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=question),
                dict(type="video", video=video_path),
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


def pointing_video_qa(video_path: str, question: str, max_tokens: int = 2048) -> tuple[str, str, list]:
    """
    Pointing Video QA: Point to objects in a video.
    
    Returns:
        Tuple of (model_response, points_text, annotated_frames)
    """
    if not video_path:
        return "请上传视频", "", []
    if not question:
        return "请输入问题", "", []
    
    model, processor = load_model()
    
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=question),
                dict(type="video", video=video_path),
            ],
        }
    ]
    
    # Use apply_chat_template which handles video processing internally
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    
    # Get video dimensions from the processed metadata
    video_metadata = inputs.pop("video_metadata", None)
    if video_metadata and len(video_metadata) > 0:
        width = getattr(video_metadata[0], "width", None) or 1920
        height = getattr(video_metadata[0], "height", None) or 1080
    else:
        # Fallback: try to get dimensions from video file
        width, height = get_video_dimensions(video_path)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Extract points
    points = extract_video_points(generated_text, image_w=width, image_h=height)
    points_str = format_points_output(points)
    
    # 提取并标注视频帧
    annotated_frames = extract_video_frames_with_points(video_path, points, max_frames=4)
    
    return generated_text, points_str, annotated_frames


def tracking_video_qa(video_path: str, question: str, max_tokens: int = 2048) -> tuple[str, str, list]:
    """
    Tracking Video QA: Track objects in a video.
    
    Returns:
        Tuple of (model_response, points_text, annotated_frames)
    """
    if not video_path:
        return "请上传视频", "", []
    if not question:
        return "请输入追踪指令", "", []
    
    model, processor = load_model()
    
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=question),
                dict(type="video", video=video_path),
            ],
        }
    ]
    
    # Use apply_chat_template which handles video processing internally
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    
    # Get video dimensions from the processed metadata
    video_metadata = inputs.pop("video_metadata", None)
    if video_metadata and len(video_metadata) > 0:
        width = getattr(video_metadata[0], "width", None) or 1920
        height = getattr(video_metadata[0], "height", None) or 1080
    else:
        # Fallback: try to get dimensions from video file
        width, height = get_video_dimensions(video_path)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Extract tracking points
    points = extract_video_points(generated_text, image_w=width, image_h=height)
    points_str = format_points_output(points)
    
    # 提取并标注视频帧
    annotated_frames = extract_video_frames_with_points(video_path, points, max_frames=4)
    
    return generated_text, points_str, annotated_frames


def multi_image_qa(images: list, question: str, max_tokens: int = 448) -> str:
    """
    Multi-image QA: Answer questions about multiple images.
    """
    if not images or len(images) == 0:
        return "Please provide at least one image."
    if not question:
        return "Please provide a question."
    
    model, processor = load_model()
    
    # Build message content
    content = [dict(type="text", text=question)]
    
    for img in images:
        if img is not None:
            if isinstance(img, str):
                img = Image.open(img)
            content.append(dict(type="image", image=img))
    
    messages = [{"role": "user", "content": content}]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


def multi_image_point_qa(images: list, question: str, max_tokens: int = 2048) -> tuple[str, str, list]:
    """
    Multi-Image Point QA: Point to objects across multiple images.
    
    Returns:
        Tuple of (model_response, points_text, annotated_images)
    """
    if not images or len(images) == 0:
        return "Please provide at least one image.", "", []
    if not question:
        return "Please provide a question.", "", []
    
    model, processor = load_model()
    
    # Build message content and collect image dimensions
    content = [dict(type="text", text=question)]
    image_widths = []
    image_heights = []
    pil_images = []
    
    for img in images:
        if img is not None:
            # 确保是 PIL Image
            if isinstance(img, str):
                img = Image.open(img)
            elif not isinstance(img, Image.Image):
                # 可能是 numpy 数组，转换为 PIL
                import numpy as np
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
            
            pil_images.append(img.copy())  # Make copies
            image_widths.append(img.width)
            image_heights.append(img.height)
            content.append(dict(type="image", image=img))
    
    if len(pil_images) == 0:
        return "No valid images provided.", "", []
    
    messages = [{"role": "user", "content": content}]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Extract points
    points = extract_multi_image_points(generated_text, image_widths, image_heights)
    points_str = format_points_output(points)
    
    # Debug: 打印点信息
    print(f"[DEBUG] 提取到 {len(points)} 个点")
    print(f"[DEBUG] 图片数量: {len(pil_images)}, 尺寸: {list(zip(image_widths, image_heights))}")
    if points:
        print(f"[DEBUG] 前3个点: {points[:3]}")
    
    # Draw points on images
    annotated_images = draw_points_on_multi_images(pil_images, points)
    
    return generated_text, points_str, annotated_images


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create the Gradio interface with multiple tabs."""
    
    # Download examples at startup
    print("Downloading example files...")
    example_files = download_examples()
    print("Examples ready.")
    
    # Get example file paths
    penguin_video = example_files["videos"].get("penguins", "")
    basketball_video = example_files["videos"].get("basketball", "")
    dog_image = example_files["images"].get("dog", "")
    cherry_image = example_files["images"].get("cherry_blossom", "")
    boat1_image = example_files["images"].get("boat1", "")
    boat2_image = example_files["images"].get("boat2", "")
    
    with gr.Blocks(title="Molmo2-8B 演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 Molmo2-8B 多模态演示
        
        本演示展示 Molmo2-8B 模型的多种能力：
        - **视频问答** - 对视频内容提问
        - **视频目标指向** - 在视频中指出特定物体
        - **视频目标追踪** - 追踪视频中的物体
        - **多图问答** - 对多张图片进行对比和问答
        - **多图目标指向** - 在多张图片中指出特定物体
        
        选择下方标签页体验不同功能！
        """)
        
        with gr.Tabs():
            # ================================================================
            # Tab 1: General Video QA
            # ================================================================
            with gr.TabItem("🎥 视频问答"):
                gr.Markdown("""
                ### 通用视频问答
                上传视频并对其内容进行提问。
                """)
                
                with gr.Row():
                    with gr.Column():
                        video_input_1 = gr.Video(label="上传视频")
                        question_1 = gr.Textbox(
                            label="问题",
                            placeholder="视频中出现了什么动物？",
                            lines=2
                        )
                        max_tokens_1 = gr.Slider(
                            minimum=64, maximum=4096, value=2048, step=64,
                            label="最大输出长度"
                        )
                        submit_btn_1 = gr.Button("提交", variant="primary")
                    
                    with gr.Column():
                        output_1 = gr.Textbox(label="回答", lines=10)
                
                gr.Examples(
                    examples=[
                        [penguin_video, "视频中出现了什么动物？"],
                        [basketball_video, "视频中在进行什么运动？"],
                    ] if penguin_video else [],
                    inputs=[video_input_1, question_1],
                    label="示例"
                )
                
                submit_btn_1.click(
                    fn=general_video_qa,
                    inputs=[video_input_1, question_1, max_tokens_1],
                    outputs=[output_1]
                )
            
            # ================================================================
            # Tab 2: Pointing Video QA
            # ================================================================
            with gr.TabItem("👆 视频目标指向"):
                gr.Markdown("""
                ### 视频目标指向
                上传视频并让模型指出特定物体的位置。
                模型会返回物体在各帧中的坐标。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_2 = gr.Video(label="上传视频")
                        question_2 = gr.Textbox(
                            label="指向指令",
                            placeholder="指出企鹅的位置",
                            lines=2
                        )
                        max_tokens_2 = gr.Slider(
                            minimum=64, maximum=4096, value=2048, step=64,
                            label="最大输出长度"
                        )
                        submit_btn_2 = gr.Button("提交", variant="primary")
                    
                    with gr.Column(scale=2):
                        output_2a = gr.Textbox(label="模型输出", lines=3)
                        output_2b = gr.Textbox(label="提取的坐标点 (帧, X, Y)", lines=6)
                        gr.Markdown("### 📌 标注后的关键帧")
                        with gr.Row():
                            output_frame_2a = gr.Image(label="帧 1", type="pil")
                            output_frame_2b = gr.Image(label="帧 2", type="pil")
                        with gr.Row():
                            output_frame_2c = gr.Image(label="帧 3", type="pil")
                            output_frame_2d = gr.Image(label="帧 4", type="pil")
                
                def pointing_video_qa_wrapper(video, question, max_tokens):
                    response, points_str, frames = pointing_video_qa(video, question, max_tokens)
                    # 填充到4帧
                    while len(frames) < 4:
                        frames.append(None)
                    return response, points_str, frames[0], frames[1], frames[2], frames[3]
                
                gr.Examples(
                    examples=[
                        [penguin_video, "Point to the penguins."],
                    ] if penguin_video else [],
                    inputs=[video_input_2, question_2],
                    label="示例"
                )
                
                submit_btn_2.click(
                    fn=pointing_video_qa_wrapper,
                    inputs=[video_input_2, question_2, max_tokens_2],
                    outputs=[output_2a, output_2b, output_frame_2a, output_frame_2b, output_frame_2c, output_frame_2d]
                )
            
            # ================================================================
            # Tab 3: Tracking Video QA
            # ================================================================
            with gr.TabItem("🎯 视频目标追踪"):
                gr.Markdown("""
                ### 视频目标追踪
                上传视频并让模型追踪特定物体。
                模型会返回物体在各帧中的追踪坐标。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_3 = gr.Video(label="上传视频")
                        question_3 = gr.Textbox(
                            label="追踪指令",
                            placeholder="追踪正在扣篮的球员",
                            lines=2
                        )
                        max_tokens_3 = gr.Slider(
                            minimum=64, maximum=4096, value=2048, step=64,
                            label="最大输出长度"
                        )
                        submit_btn_3 = gr.Button("提交", variant="primary")
                    
                    with gr.Column(scale=2):
                        output_3a = gr.Textbox(label="模型输出", lines=3)
                        output_3b = gr.Textbox(label="追踪坐标点 (帧, X, Y)", lines=6)
                        gr.Markdown("### 📌 标注后的关键帧")
                        with gr.Row():
                            output_frame_3a = gr.Image(label="帧 1", type="pil")
                            output_frame_3b = gr.Image(label="帧 2", type="pil")
                        with gr.Row():
                            output_frame_3c = gr.Image(label="帧 3", type="pil")
                            output_frame_3d = gr.Image(label="帧 4", type="pil")
                
                def tracking_video_qa_wrapper(video, question, max_tokens):
                    response, points_str, frames = tracking_video_qa(video, question, max_tokens)
                    # 填充到4帧
                    while len(frames) < 4:
                        frames.append(None)
                    return response, points_str, frames[0], frames[1], frames[2], frames[3]
                
                gr.Examples(
                    examples=[
                        [basketball_video, "Track the player who is dunking"],
                    ] if basketball_video else [],
                    inputs=[video_input_3, question_3],
                    label="示例"
                )
                
                submit_btn_3.click(
                    fn=tracking_video_qa_wrapper,
                    inputs=[video_input_3, question_3, max_tokens_3],
                    outputs=[output_3a, output_3b, output_frame_3a, output_frame_3b, output_frame_3c, output_frame_3d]
                )
            
            # ================================================================
            # Tab 4: Multi-Image QA
            # ================================================================
            with gr.TabItem("🖼️ 图片问答"):
                gr.Markdown("""
                ### 图片问答
                上传图片并对其内容进行提问。
                """)
                
                with gr.Row():
                    with gr.Column():
                        image_input_4a = gr.Image(label="图片 1", type="pil")
                        image_input_4b = gr.Image(label="图片 2", type="pil")
                        question_4 = gr.Textbox(
                            label="问题",
                            placeholder="对比这两张图片",
                            lines=2
                        )
                        max_tokens_4 = gr.Slider(
                            minimum=64, maximum=2048, value=448, step=64,
                            label="最大输出长度"
                        )
                        submit_btn_4 = gr.Button("提交", variant="primary")
                    
                    with gr.Column():
                        output_4 = gr.Textbox(label="回答", lines=15)
                
                def multi_image_qa_wrapper(img1, img2, question, max_tokens):
                    images = [img for img in [img1, img2] if img is not None]
                    return multi_image_qa(images, question, max_tokens)
                
                gr.Examples(
                    examples=[
                        [dog_image, None, "Describe this image."],
                        [cherry_image, None, "What do you see in this image?"],
                    ] if dog_image else [],
                    inputs=[image_input_4a, image_input_4b, question_4],
                    label="示例"
                )
                
                submit_btn_4.click(
                    fn=multi_image_qa_wrapper,
                    inputs=[image_input_4a, image_input_4b, question_4, max_tokens_4],
                    outputs=[output_4]
                )
            
            # ================================================================
            # Tab 5: Multi-Image Point QA
            # ================================================================
            with gr.TabItem("📍 图片标记"):
                gr.Markdown("""
                ### 图片标记
                上传图片并让模型指出特定物体的位置。
                标记点会直接显示在图片上。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input_5a = gr.Image(label="图片 1", type="pil")
                        image_input_5b = gr.Image(label="图片 2（可选）", type="pil")
                        question_5 = gr.Textbox(
                            label="指向指令",
                            placeholder="Point to the boats",
                            lines=2
                        )
                        max_tokens_5 = gr.Slider(
                            minimum=64, maximum=4096, value=2048, step=64,
                            label="最大输出长度"
                        )
                        submit_btn_5 = gr.Button("提交", variant="primary")
                    
                    with gr.Column(scale=2):
                        output_5a = gr.Textbox(label="模型输出", lines=3)
                        output_5b = gr.Textbox(label="提取的坐标点 (图片, X, Y)", lines=6)
                        gr.Markdown("### 📌 标注后的图片")
                        with gr.Row():
                            output_img_5a = gr.Image(label="图片 1 - 标注", type="pil")
                            output_img_5b = gr.Image(label="图片 2 - 标注", type="pil")
                
                def multi_image_point_qa_wrapper(img1, img2, question, max_tokens):
                    images = [img for img in [img1, img2] if img is not None]
                    response, points_str, annotated_images = multi_image_point_qa(images, question, max_tokens)
                    
                    # Prepare output images (handle cases with 1 or 2 images)
                    out_img1 = annotated_images[0] if len(annotated_images) > 0 else None
                    out_img2 = annotated_images[1] if len(annotated_images) > 1 else None
                    
                    return response, points_str, out_img1, out_img2
                
                gr.Examples(
                    examples=[
                        [boat1_image, None, "Point to the boats"],
                        [dog_image, None, "Point to the dog's eyes"],
                        [cherry_image, None, "Point to the flowers"],
                    ] if boat1_image else [],
                    inputs=[image_input_5a, image_input_5b, question_5],
                    label="示例"
                )
                
                submit_btn_5.click(
                    fn=multi_image_point_qa_wrapper,
                    inputs=[image_input_5a, image_input_5b, question_5, max_tokens_5],
                    outputs=[output_5a, output_5b, output_img_5a, output_img_5b]
                )
        
        gr.Markdown("""
        ---
        **注意：** 模型会在首次使用时加载，可能需要一些时间。后续查询会更快。
        
        **模型：** Allen AI 的 Molmo2-8B
        """)
    
    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Pre-load model (optional, can be loaded lazily)
    print("Initializing Molmo2 Demo...")
    
    demo = create_interface()
    
    # Launch the Gradio app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
