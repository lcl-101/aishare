import argparse
import math
import os
import subprocess
from datetime import datetime
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from diffsynth import ModelManager, WanVideoPipeline
from diffsynth.data import save_video
from diffsynth.models.camer import CameraDemo
from diffsynth.models.face_align import FaceAlignment
from diffsynth.models.pdf import (FanEncoder, det_landmarks,
                                  get_drive_expression_pd_fgc)
from diffsynth.pipelines.wan_video import PortraitAdapter
from utils import merge_audio_to_video


# 导入必要的多人处理函数
import torch.nn.functional as F

# 默认参数配置
DEFAULT_CONFIG = {
    "portrait_checkpoint": "./checkpoints/FantasyPortrait/fantasyportrait_model.ckpt",
    "alignment_model_path": "./checkpoints/FantasyPortrait/face_landmark.onnx",
    "det_model_path": "./checkpoints/FantasyPortrait/face_det.onnx",
    "pd_fpg_model_path": "./checkpoints/FantasyPortrait/pd_fpg.pth",
    "wan_model_path": "./checkpoints/Wan2.1-I2V-14B-720P",
    "output_path": "./outputs/",
    "height": 480,
    "width": 832,
    "portrait_scale": 1.0,
    "cfg_scale": 1.0,
    "portrait_cfg_scale": 4.0,
    "scale_image": True,
    "portrait_in_dim": 768,
    "portrait_proj_dim": 2048,
    "num_frames": 201,
    "seed": 42,
    "max_size": 720,
    "fps": 25,  # 新增 fps 参数用于多人处理
}

# 全局变量
pipe = None
face_aligner = None
pd_fpg_motion = None
portrait_model = None
device = torch.device("cuda")


def resize_mask(mask):
    """
    Downsample the mask both temporally and spatially to match the size of the video compressed by VAE.
    """
    f, h, w = mask.shape

    first_frame = mask[0].unsqueeze(0).unsqueeze(0)
    first_frame = F.max_pool2d(first_frame, kernel_size=16, stride=16)
    first_frame = first_frame.squeeze(0).squeeze(0)

    mask_rest = mask[1:].unsqueeze(0).unsqueeze(0)
    mask_rest = F.max_pool3d(mask_rest, kernel_size=(4, 16, 16), stride=(4, 16, 16))
    mask_rest = mask_rest.squeeze(0).squeeze(0)
    mask_resized = torch.cat([first_frame.unsqueeze(0), mask_rest], dim=0)

    return mask_resized


def build_attn_mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: shape [B, L1]
    b: shape [B, L2]
    return: attn_mask, shape [B, L1, L2], dtype=torch.bool or torch.int
    """
    attn_mask = a.unsqueeze(2) == b.unsqueeze(1)
    return ~attn_mask


def compute_max_xy(rect_list):
    x1_ref, y1_ref, x2_ref, y2_ref = rect_list[0]
    cx_ref = (x1_ref + x2_ref) / 2
    cy_ref = (y1_ref + y2_ref) / 2
    width_ref = x2_ref - x1_ref
    height_ref = y2_ref - y1_ref

    rect_array = np.array(rect_list)
    max_values = np.max(rect_array, axis=0)
    min_values = np.min(rect_array, axis=0)

    max_x1, max_y1, max_x2, max_y2 = max_values
    min_x1, min_y1, min_x2, min_y2 = min_values

    max_rect = [min_x1, min_y1, max_x2, max_y2]

    max_left_move = (x1_ref - min_x1) / width_ref
    max_right_move = (max_x2 - x2_ref) / width_ref
    max_up_move = (y1_ref - min_y1) / height_ref
    max_down_move = (max_y2 - y2_ref) / height_ref
    return (
        min(max_left_move, 0.25),
        min(max_right_move, 0.25),
        min(max_up_move, 0.25),
        min(max_down_move, 0.25),
    )


def create_mask(image, bounding_boxes, proj_split, video_rect_list):
    """
    Computes the maximum range of facial movements from the face regions in the driving video to obtain the corresponding face mask for the reference image.
    """
    width, height = image.size
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[0] + box[2]) / 2)

    num_faces = len(bounding_boxes)
    assert proj_split.shape[2] % num_faces == 0

    mask = torch.zeros((proj_split.size(1) - 1) * 4 + 1, height, width, device="cuda")

    adapter_mask = torch.zeros(proj_split.squeeze(0).shape[:-1], device="cuda")
    f, l = adapter_mask.shape

    extend_bounding_boxes = []

    for i, (face_rect, video_rect) in enumerate(zip(bounding_boxes, video_rect_list)):
        max_left_move, max_right_move, max_up_move, max_down_move = compute_max_xy(
            video_rect
        )

        x1, y1, x2, y2 = face_rect
        width_face, height_face = int(x2 - x1), int(y2 - y1)
        x1 -= width_face * max_left_move
        x2 += width_face * max_right_move
        y1 -= height_face * max_up_move
        y2 += height_face * max_down_move

        x1 = max(0, int(x1))
        x2 = min(width, int(x2))
        y1 = max(0, int(y1))
        y2 = min(height, int(y2))

        extend_bounding_boxes.append([x1, y1, x2, y2])

        mask[:, y1:y2, x1:x2] = i + 1

        adapter_face_index_begin = (l // num_faces) * i
        adapter_face_index_end = (l // num_faces) * (i + 1)

        adapter_mask[:, adapter_face_index_begin:adapter_face_index_end] = i + 1

    mask_latents = resize_mask(mask)
    _, l_h, l_w = mask_latents.shape[:3]

    mask_latents = mask_latents.view(f, -1)
    attn_mask = build_attn_mask(mask_latents, adapter_mask)

    return attn_mask, extend_bounding_boxes


def get_emo_feature_multi(
    video_path, face_aligner, pd_fpg_motion, num_frames, device=torch.device("cuda")
):
    """多人处理版本的表情特征提取"""
    pd_fpg_motion = pd_fpg_motion.to(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []
    ret, frame = cap.read()
    while ret:
        frame_list.append(frame.copy())
        ret, frame = cap.read()
    cap.release()

    num_frames = min(len(frame_list), num_frames)
    num_frames = find_replacement(num_frames)
    frame_list = frame_list[:num_frames]

    landmark_list, rect_list = det_landmarks(face_aligner, frame_list)[1:]

    emo_list = get_drive_expression_pd_fgc(
        pd_fpg_motion, frame_list, landmark_list, device
    )

    emo_feat_list = []
    head_emo_feat_list = []
    for emo in emo_list:
        headpose_emb = emo["headpose_emb"]
        eye_embed = emo["eye_embed"]
        emo_embed = emo["emo_embed"]
        mouth_feat = emo["mouth_feat"]

        emo_feat = torch.cat([eye_embed, emo_embed, mouth_feat], dim=1)
        head_emo_feat = torch.cat([headpose_emb, emo_feat], dim=1)

        emo_feat_list.append(emo_feat)
        head_emo_feat_list.append(head_emo_feat)

    emo_feat_all = torch.cat(emo_feat_list, dim=0)
    head_emo_feat_all = torch.cat(head_emo_feat_list, dim=0)

    return emo_feat_all, head_emo_feat_all, fps, num_frames, rect_list, frame_list


def process_multi_video(portrait_model, face_aligner, pd_fpg_motion, image, video_list, num_frames):
    """
    处理多人视频生成
    """
    width, height = image.size
    bounding_boxes, _, score = face_aligner.face_alignment_module.face_detector.detect(
        np.array(image)[:, :, ::-1]
    )
    num_faces = len(bounding_boxes)

    if num_faces <= 1:
        raise ValueError(f"图像中检测到的人脸数量 {num_faces} 小于等于1。多人模式需要至少2个人脸。")
    if len(video_list) > num_faces:
        video_list = video_list[:num_faces]

    if len(video_list) != num_faces:
        raise ValueError(
            f"视频数量 {len(video_list)} 与人脸数量 {num_faces} 不匹配！"
        )

    face_motion_feat = []
    num_frames_list = []
    video_rect_list = []
    frame_list_list = []
    
    with torch.no_grad():
        for video_path in video_list:
            (
                emo_feat_all,
                head_emo_feat_all,
                fps,
                actual_num_frames,
                rect_list,
                frame_list,
            ) = get_emo_feature_multi(video_path, face_aligner, pd_fpg_motion, num_frames)
            face_motion_feat.append(head_emo_feat_all)
            num_frames_list.append(actual_num_frames)
            video_rect_list.append(rect_list)
            frame_list_list.append(frame_list)

    num_frames = min(num_frames_list)
    face_motion_feat = [i[:num_frames, :].unsqueeze(0) for i in face_motion_feat]
    video_rect_list = [i[:num_frames] for i in video_rect_list]

    proj_split = []
    for face_motion_feat_ in face_motion_feat:
        adapter_proj = portrait_model.get_adapter_proj(face_motion_feat_.to("cuda"))
        pos_idx_range = portrait_model.split_audio_adapter_sequence(
            adapter_proj.size(1), num_frames=num_frames
        )
        proj_split_, adapter_context_lens = portrait_model.split_tensor_with_padding(
            adapter_proj, pos_idx_range, expand_length=0
        )
        proj_split.append(proj_split_)

    proj_split = torch.cat(proj_split[::-1], dim=-2)

    adapter_attn_mask, extend_bounding_boxes = create_mask(
        image, bounding_boxes, proj_split, video_rect_list
    )

    return (
        proj_split,
        adapter_attn_mask,
        extend_bounding_boxes,
        frame_list_list,
        num_frames,
        video_list,
        fps
    )


def find_replacement(a):
    while a > 0:
        if (a - 1) % 4 == 0:
            return a
        a -= 1
    return 0


def get_emo_feature(
    video_path, face_aligner, pd_fpg_motion, num_frames, device=torch.device("cuda")
):
    pd_fpg_motion = pd_fpg_motion.to(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []
    ret, frame = cap.read()
    while ret:
        resized_frame = frame
        frame_list.append(resized_frame.copy())
        ret, frame = cap.read()
    cap.release()

    num_frames = min(len(frame_list), num_frames)
    num_frames = find_replacement(num_frames)
    frame_list = frame_list[:num_frames]

    landmark_list = det_landmarks(face_aligner, frame_list)[1]
    emo_list = get_drive_expression_pd_fgc(
        pd_fpg_motion, frame_list, landmark_list, device
    )

    emo_feat_list = []
    head_emo_feat_list = []
    for emo in emo_list:
        headpose_emb = emo["headpose_emb"]
        eye_embed = emo["eye_embed"]
        emo_embed = emo["emo_embed"]
        mouth_feat = emo["mouth_feat"]

        emo_feat = torch.cat([eye_embed, emo_embed, mouth_feat], dim=1)
        head_emo_feat = torch.cat([headpose_emb, emo_feat], dim=1)

        emo_feat_list.append(emo_feat)
        head_emo_feat_list.append(head_emo_feat)

    emo_feat_all = torch.cat(emo_feat_list, dim=0)
    head_emo_feat_all = torch.cat(head_emo_feat_list, dim=0)

    return emo_feat_all, head_emo_feat_all, fps, num_frames


def load_wan_video():
    """加载 WAN Video 模型"""
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                os.path.join(
                    DEFAULT_CONFIG["wan_model_path"],
                    "diffusion_pytorch_model-00001-of-00007.safetensors",
                ),
                os.path.join(
                    DEFAULT_CONFIG["wan_model_path"],
                    "diffusion_pytorch_model-00002-of-00007.safetensors",
                ),
                os.path.join(
                    DEFAULT_CONFIG["wan_model_path"],
                    "diffusion_pytorch_model-00003-of-00007.safetensors",
                ),
                os.path.join(
                    DEFAULT_CONFIG["wan_model_path"],
                    "diffusion_pytorch_model-00004-of-00007.safetensors",
                ),
                os.path.join(
                    DEFAULT_CONFIG["wan_model_path"],
                    "diffusion_pytorch_model-00005-of-00007.safetensors",
                ),
                os.path.join(
                    DEFAULT_CONFIG["wan_model_path"],
                    "diffusion_pytorch_model-00006-of-00007.safetensors",
                ),
                os.path.join(
                    DEFAULT_CONFIG["wan_model_path"],
                    "diffusion_pytorch_model-00007-of-00007.safetensors",
                ),
            ],
            os.path.join(
                DEFAULT_CONFIG["wan_model_path"],
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
            os.path.join(DEFAULT_CONFIG["wan_model_path"], "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(DEFAULT_CONFIG["wan_model_path"], "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    return pipe


def load_pd_fgc_model():
    """加载 PD-FGC 模型"""
    face_aligner = CameraDemo(
        face_alignment_module=FaceAlignment(
            gpu_id=None,
            alignment_model_path=DEFAULT_CONFIG["alignment_model_path"],
            det_model_path=DEFAULT_CONFIG["det_model_path"],
        ),
        reset=False,
    )

    pd_fpg_motion = FanEncoder()
    pd_fpg_checkpoint = torch.load(DEFAULT_CONFIG["pd_fpg_model_path"], map_location="cpu")
    m, u = pd_fpg_motion.load_state_dict(pd_fpg_checkpoint, strict=False)
    pd_fpg_motion = pd_fpg_motion.eval()

    return face_aligner, pd_fpg_motion


def initialize_models():
    """初始化所有模型"""
    global pipe, face_aligner, pd_fpg_motion, portrait_model
    
    print("正在加载模型，请稍候...")
    
    # 创建输出目录
    os.makedirs(DEFAULT_CONFIG["output_path"], exist_ok=True)
    
    # 加载模型
    pipe = load_wan_video()
    face_aligner, pd_fpg_motion = load_pd_fgc_model()
    
    portrait_model = PortraitAdapter(
        pipe.dit, DEFAULT_CONFIG["portrait_in_dim"], DEFAULT_CONFIG["portrait_proj_dim"]
    ).to("cuda")
    portrait_model.load_portrait_adapter(DEFAULT_CONFIG["portrait_checkpoint"], pipe.dit)
    pipe.dit.to("cuda")
    
    print(f"FantasyPortrait 模型加载完成：{DEFAULT_CONFIG['portrait_checkpoint']}")
    return "模型加载完成！"


def generate_multi_video(
    input_image,
    driven_videos,
    prompt,
    num_frames,
    cfg_scale,
    portrait_scale,
    portrait_cfg_scale,
    seed,
    max_size,
    scale_image,
    fps,
    progress=gr.Progress()
):
    """生成多人视频的主函数"""
    global pipe, face_aligner, pd_fpg_motion, portrait_model
    
    if pipe is None or face_aligner is None or pd_fpg_motion is None or portrait_model is None:
        return None, "请先初始化模型！"
    
    try:
        progress(0, desc="开始处理...")
        
        # 处理输入图像
        if input_image is None:
            return None, "请上传输入图像！"
        
        if driven_videos is None or len(driven_videos) == 0:
            return None, "请上传至少一个驱动视频！"
        
        progress(0.1, desc="处理输入图像...")
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            input_image.save(tmp_img.name)
            input_image_path = tmp_img.name
        
        # 处理驱动视频列表
        video_paths = []
        for video in driven_videos:
            if isinstance(video, str):
                # 如果是文件路径
                video_paths.append(video)
            else:
                # 如果是上传的文件对象
                video_path = video.name if hasattr(video, 'name') else video
                video_paths.append(video_path)
        
        # 加载和处理图像
        image = Image.open(input_image_path).convert("RGB")
        width, height = image.size
        if scale_image:
            scale = max_size / max(width, height)
            width, height = (int(width * scale), int(height * scale))
            image = image.resize([width, height], Image.LANCZOS)
        
        # 调整尺寸以符合模型要求
        height, width = pipe.check_resize_height_width(height, width)
        image = image.resize([width, height], Image.LANCZOS)
        
        progress(0.3, desc="处理多人表情特征...")
        
        # 处理多人视频
        (
            proj_split,
            adapter_attn_mask,
            extend_bounding_boxes,
            frame_list_list,
            actual_num_frames,
            video_list_sample,
            video_fps
        ) = process_multi_video(
            portrait_model, face_aligner, pd_fpg_motion, image, video_paths, num_frames
        )
        
        progress(0.7, desc="生成视频...")
        
        negative_prompt = "人物嘴巴不停地说话，人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        video_audio = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=image,
            width=width,
            height=height,
            num_frames=actual_num_frames,
            num_inference_steps=30,
            seed=seed,
            tiled=True,
            ip_scale=portrait_scale,
            cfg_scale=cfg_scale,
            ip_cfg_scale=portrait_cfg_scale,
            adapter_proj=proj_split,
            adapter_context_lens=None,
            latents_num_frames=(actual_num_frames - 1) // 4 + 1,
            adapter_attn_mask=adapter_attn_mask,
        )
        
        progress(0.9, desc="保存视频...")
        
        # 生成保存路径
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        save_name = f"{timestamp_str}_multi_generated"
        save_video_path = os.path.join(DEFAULT_CONFIG["output_path"], f"{save_name}.mp4")
        
        # 保存视频
        save_video(
            video_audio, save_video_path, fps=fps, quality=5
        )
        
        # 清理临时文件
        os.unlink(input_image_path)
        
        progress(1.0, desc="完成！")
        
        return save_video_path, f"多人视频生成成功！检测到 {len(extend_bounding_boxes)} 个人脸。"
        
    except Exception as e:
        return None, f"生成失败：{str(e)}"


def load_multi_example_files():
    """加载多人示例文件"""
    try:
        example_image_path = "./assert/two_person.png"
        # 注意：根据 infer_multi_diff.sh，这里应该使用正确的路径
        example_video_paths = ["./assert/trump.mp4", "./assert/biden.mp4"]
        
        # 检查文件是否存在
        if not os.path.exists(example_image_path):
            return None, [], [], "示例图片文件不存在"
        
        existing_videos = []
        for video_path in example_video_paths:
            if os.path.exists(video_path):
                existing_videos.append(video_path)
        
        if len(existing_videos) == 0:
            return None, [], [], "示例视频文件都不存在"
        
        # 加载图片
        example_image = Image.open(example_image_path)
        
        # 返回图片和视频文件路径列表
        return example_image, existing_videos, existing_videos, f"多人示例文件加载成功！共加载 {len(existing_videos)} 个视频。"
    except Exception as e:
        return None, [], [], f"加载多人示例文件失败：{str(e)}"


def generate_video(
    input_image,
    driven_video,
    prompt,
    num_frames,
    cfg_scale,
    portrait_scale,
    portrait_cfg_scale,
    seed,
    max_size,
    scale_image,
    progress=gr.Progress()
):
    """生成视频的主函数"""
    global pipe, face_aligner, pd_fpg_motion, portrait_model
    
    if pipe is None or face_aligner is None or pd_fpg_motion is None or portrait_model is None:
        return None, "请先初始化模型！"
    
    try:
        progress(0, desc="开始处理...")
        
        # 处理输入图像
        if input_image is None:
            return None, "请上传输入图像！"
        
        if driven_video is None:
            return None, "请上传驱动视频！"
        
        progress(0.1, desc="处理输入图像...")
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            input_image.save(tmp_img.name)
            input_image_path = tmp_img.name
        
        # 处理驱动视频
        if isinstance(driven_video, str):
            # 如果是文件路径
            driven_video_path = driven_video
            is_temp_video = False
        else:
            # 如果是上传的文件对象
            driven_video_path = driven_video.name if hasattr(driven_video, 'name') else driven_video
            is_temp_video = False
        
        # 加载和处理图像
        image = Image.open(input_image_path).convert("RGB")
        width, height = image.size
        if scale_image:
            scale = max_size / max(width, height)
            width, height = (int(width * scale), int(height * scale))
            image = image.resize([width, height], Image.LANCZOS)
        
        progress(0.3, desc="提取表情特征...")
        
        # 提取表情特征
        with torch.no_grad():
            emo_feat_all, head_emo_feat_all, fps, actual_num_frames = get_emo_feature(
                driven_video_path, face_aligner, pd_fpg_motion, num_frames
            )
        
        emo_feat_all, head_emo_feat_all = emo_feat_all.unsqueeze(
            0
        ), head_emo_feat_all.unsqueeze(0)
        
        progress(0.5, desc="准备适配器...")
        
        adapter_proj = portrait_model.get_adapter_proj(head_emo_feat_all.to(device))
        pos_idx_range = portrait_model.split_audio_adapter_sequence(
            adapter_proj.size(1), num_frames=actual_num_frames
        )
        proj_split, context_lens = portrait_model.split_tensor_with_padding(
            adapter_proj, pos_idx_range, expand_length=0
        )
        
        progress(0.7, desc="生成视频...")
        
        negative_prompt = "人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        video_audio = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=image,
            width=width,
            height=height,
            num_frames=actual_num_frames,
            num_inference_steps=30,
            seed=seed,
            tiled=True,
            ip_scale=portrait_scale,
            cfg_scale=cfg_scale,
            ip_cfg_scale=portrait_cfg_scale,
            adapter_proj=proj_split,
            adapter_context_lens=context_lens,
            latents_num_frames=(actual_num_frames - 1) // 4 + 1,
        )
        
        progress(0.9, desc="保存视频...")
        
        # 生成保存路径
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        save_name = f"{timestamp_str}_generated"
        save_video_path = os.path.join(DEFAULT_CONFIG["output_path"], f"{save_name}.mp4")
        
        # 保存视频
        save_video(
            video_audio, save_video_path, fps=fps, quality=5
        )
        
        progress(0.95, desc="添加音频...")
        
        # 添加音频
        save_video_path_with_audio = os.path.join(
            DEFAULT_CONFIG["output_path"], f"{save_name}_with_audio.mp4"
        )
        merge_audio_to_video(
            driven_video_path, save_video_path, save_video_path_with_audio
        )
        
        # 清理临时文件
        os.unlink(input_image_path)
        # 注意：不需要删除视频文件，因为 Gradio 会自动管理
        
        progress(1.0, desc="完成！")
        
        return save_video_path_with_audio, "视频生成成功！"
        
    except Exception as e:
        return None, f"生成失败：{str(e)}"


def update_video_preview(video_file):
    """更新视频预览"""
    if video_file is None:
        return None
    return video_file


def load_example_files():
    """加载示例文件"""
    try:
        example_image_path = "./assert/man.jpeg"
        example_video_path = "./assert/jgz.mp4"
        
        # 检查文件是否存在
        if not os.path.exists(example_image_path):
            return None, None, None, "示例图片文件不存在"
        if not os.path.exists(example_video_path):
            return None, None, None, "示例视频文件不存在"
        
        # 加载图片
        example_image = Image.open(example_image_path)
        
        # 返回图片、视频文件路径和视频预览路径
        return example_image, example_video_path, example_video_path, "示例文件加载成功！"
    except Exception as e:
        return None, None, None, f"加载示例文件失败：{str(e)}"


# 创建 Gradio 界面
def create_interface():
    with gr.Blocks(title="FantasyPortrait WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# FantasyPortrait WebUI")
        gr.Markdown("基于 FantasyPortrait 的肖像视频生成工具")
        
        # 模型初始化部分
        with gr.Row():
            init_btn = gr.Button("初始化模型", variant="primary", size="lg")
            init_status = gr.Textbox(label="初始化状态", interactive=False, scale=2)
        
        # 创建标签页
        with gr.Tabs():
            # 单人处理标签页
            with gr.TabItem("单人肖像视频生成"):
                with gr.Row():
                    # 左侧：输入和参数
                    with gr.Column(scale=1):
                        gr.Markdown("## 输入设置")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                input_image = gr.Image(
                                    label="输入肖像图片",
                                    type="pil",
                                    format="jpg",
                                    height=300
                                )
                            
                            with gr.Column(scale=1):
                                driven_video = gr.File(
                                    label="驱动视频",
                                    file_types=[".mp4", ".avi", ".mov"],
                                    file_count="single"
                                )
                                # 添加视频预览
                                driven_video_preview = gr.Video(
                                    label="视频预览",
                                    height=300,
                                    interactive=False
                                )
                        
                        prompt = gr.Textbox(
                            label="提示词",
                            placeholder="输入生成提示词（可选）",
                            value="",
                            lines=2
                        )
                        
                        gr.Markdown("## 生成参数")
                        
                        with gr.Row():
                            num_frames = gr.Slider(
                                label="帧数",
                                minimum=50,
                                maximum=300,
                                value=DEFAULT_CONFIG["num_frames"],
                                step=1
                            )
                            
                            seed = gr.Slider(
                                label="随机种子",
                                minimum=0,
                                maximum=999999,
                                value=DEFAULT_CONFIG["seed"],
                                step=1
                            )
                        
                        with gr.Row():
                            cfg_scale = gr.Slider(
                                label="CFG Scale",
                                minimum=0.1,
                                maximum=20.0,
                                value=DEFAULT_CONFIG["cfg_scale"],
                                step=0.1
                            )
                            
                            portrait_scale = gr.Slider(
                                label="Portrait Scale",
                                minimum=0.1,
                                maximum=5.0,
                                value=DEFAULT_CONFIG["portrait_scale"],
                                step=0.1
                            )
                        
                        with gr.Row():
                            portrait_cfg_scale = gr.Slider(
                                label="Portrait CFG Scale",
                                minimum=0.1,
                                maximum=20.0,
                                value=DEFAULT_CONFIG["portrait_cfg_scale"],
                                step=0.1
                            )
                            
                            max_size = gr.Slider(
                                label="最大尺寸",
                                minimum=256,
                                maximum=1024,
                                value=DEFAULT_CONFIG["max_size"],
                                step=64
                            )
                        
                        scale_image = gr.Checkbox(
                            label="缩放图像",
                            value=DEFAULT_CONFIG["scale_image"]
                        )
                        
                        generate_btn = gr.Button("生成视频", variant="primary", size="lg")
                        
                    # 右侧：输出结果
                    with gr.Column(scale=1):
                        gr.Markdown("## 生成结果")
                        output_video = gr.Video(label="输出视频", height=400)
                        output_status = gr.Textbox(label="生成状态", interactive=False)
                
                # 单人示例部分
                gr.Markdown("---")
                gr.Markdown("## 快速开始示例")
                
                with gr.Row():
                    load_example_btn = gr.Button("加载示例文件 (man.jpeg + jgz.mp4)", variant="secondary", size="lg")
            
            # 多人处理标签页
            with gr.TabItem("多人肖像视频生成"):
                with gr.Row():
                    # 左侧：输入和参数
                    with gr.Column(scale=1):
                        gr.Markdown("## 输入设置")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                multi_input_image = gr.Image(
                                    label="输入多人肖像图片",
                                    type="pil",
                                    format="jpg",
                                    height=300
                                )
                            
                            with gr.Column(scale=1):
                                multi_driven_videos = gr.File(
                                    label="驱动视频（上传多个视频文件）",
                                    file_types=[".mp4", ".avi", ".mov"],
                                    file_count="multiple"
                                )
                                # 添加视频预览
                                multi_driven_videos_preview = gr.Gallery(
                                    label="视频预览",
                                    height=300,
                                    show_label=True,
                                    columns=2
                                )
                        
                        multi_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="输入生成提示词（可选）",
                            value="",
                            lines=2
                        )
                        
                        gr.Markdown("## 生成参数")
                        
                        with gr.Row():
                            multi_num_frames = gr.Slider(
                                label="帧数",
                                minimum=50,
                                maximum=300,
                                value=DEFAULT_CONFIG["num_frames"],
                                step=1
                            )
                            
                            multi_seed = gr.Slider(
                                label="随机种子",
                                minimum=0,
                                maximum=999999,
                                value=DEFAULT_CONFIG["seed"],
                                step=1
                            )
                        
                        with gr.Row():
                            multi_cfg_scale = gr.Slider(
                                label="CFG Scale",
                                minimum=0.1,
                                maximum=20.0,
                                value=DEFAULT_CONFIG["cfg_scale"],
                                step=0.1
                            )
                            
                            multi_portrait_scale = gr.Slider(
                                label="Portrait Scale",
                                minimum=0.1,
                                maximum=5.0,
                                value=DEFAULT_CONFIG["portrait_scale"],
                                step=0.1
                            )
                        
                        with gr.Row():
                            multi_portrait_cfg_scale = gr.Slider(
                                label="Portrait CFG Scale",
                                minimum=0.1,
                                maximum=20.0,
                                value=DEFAULT_CONFIG["portrait_cfg_scale"],
                                step=0.1
                            )
                            
                            multi_max_size = gr.Slider(
                                label="最大尺寸",
                                minimum=256,
                                maximum=1024,
                                value=DEFAULT_CONFIG["max_size"],
                                step=64
                            )
                        
                        with gr.Row():
                            multi_scale_image = gr.Checkbox(
                                label="缩放图像",
                                value=DEFAULT_CONFIG["scale_image"]
                            )
                            
                            multi_fps = gr.Slider(
                                label="视频帧率",
                                minimum=10,
                                maximum=60,
                                value=DEFAULT_CONFIG["fps"],
                                step=1
                            )
                        
                        multi_generate_btn = gr.Button("生成多人视频", variant="primary", size="lg")
                        
                    # 右侧：输出结果
                    with gr.Column(scale=1):
                        gr.Markdown("## 生成结果")
                        multi_output_video = gr.Video(label="输出视频", height=400)
                        multi_output_status = gr.Textbox(label="生成状态", interactive=False)
                        
                        # 添加多人处理说明
                        gr.Markdown("""
                        ### 多人处理说明：
                        - 上传的图片应包含多个人脸
                        - 需要为每个人脸上传对应的驱动视频
                        - 系统会自动检测人脸数量并匹配视频
                        - 视频数量应与检测到的人脸数量一致
                        """)
                
                # 多人示例部分
                gr.Markdown("---")
                gr.Markdown("## 多人快速开始示例")
                
                with gr.Row():
                    load_multi_example_btn = gr.Button("加载多人示例文件 (two_person.png + trump.mp4 + biden.mp4)", variant="secondary", size="lg")
        
        # 使用说明（在标签页外部）
        gr.Markdown("---")
        with gr.Row():
            gr.Markdown("""
            ### 使用步骤：
            1. **初始化模型**：点击 "初始化模型" 按钮加载所有必要的模型
            2. **选择模式**：选择 "单人肖像视频生成" 或 "多人肖像视频生成" 标签页
            3. **加载文件**：点击对应的示例按钮或手动上传图片和视频
            4. **调整参数**：根据需要调整生成参数（可选）
            5. **生成视频**：点击对应的生成按钮开始处理
            """)
        
        # 绑定事件
        init_btn.click(
            fn=initialize_models,
            outputs=init_status
        )
        
        # 单人模式事件绑定
        driven_video.change(
            fn=update_video_preview,
            inputs=driven_video,
            outputs=driven_video_preview
        )
        
        load_example_btn.click(
            fn=load_example_files,
            outputs=[input_image, driven_video, driven_video_preview, output_status]
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                input_image,
                driven_video,
                prompt,
                num_frames,
                cfg_scale,
                portrait_scale,
                portrait_cfg_scale,
                seed,
                max_size,
                scale_image
            ],
            outputs=[output_video, output_status]
        )
        
        # 多人模式事件绑定
        load_multi_example_btn.click(
            fn=load_multi_example_files,
            outputs=[multi_input_image, multi_driven_videos, multi_driven_videos_preview, multi_output_status]
        )
        
        multi_generate_btn.click(
            fn=generate_multi_video,
            inputs=[
                multi_input_image,
                multi_driven_videos,
                multi_prompt,
                multi_num_frames,
                multi_cfg_scale,
                multi_portrait_scale,
                multi_portrait_cfg_scale,
                multi_seed,
                multi_max_size,
                multi_scale_image,
                multi_fps
            ],
            outputs=[multi_output_video, multi_output_status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
