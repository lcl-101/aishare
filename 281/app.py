#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StoryMem Gradio Web 界面
基于 StoryMem 的故事视频生成工具
"""

import os
import gc
import sys
import glob
import json5
import logging
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import gradio as gr
from PIL import Image
import numpy as np

# 内存优化设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 项目路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video

# ==================== 全局配置 ====================
# 模型路径配置
T2V_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "Wan2.2-T2V-A14B"
I2V_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "Wan2.2-I2V-A14B"
LORA_MI2V_PATH = PROJECT_ROOT / "checkpoints" / "StoryMem" / "Wan2.2-MI2V-A14B"
LORA_MM2V_PATH = PROJECT_ROOT / "checkpoints" / "StoryMem" / "Wan2.2-MM2V-A14B"
STORY_DIR = PROJECT_ROOT / "story"
OUTPUT_DIR = PROJECT_ROOT / "results"

# 默认参数
DEFAULT_SIZE = "832*480"
DEFAULT_MAX_MEMORY_SIZE = 10
DEFAULT_SEED = 0
DEFAULT_SAMPLE_GUIDE_SCALE = 3.5
DEFAULT_LORA_RANK = 128
DEFAULT_FRAME_NUM = 81  # 帧数 (帧数必须是 4n+1)，81帧约5秒

# 显存充足时可以禁用 offload 以提升速度
# H20 141GB 显存足够运行单个模型不需要 offload
USE_OFFLOAD_MODEL = False  # 设为 False 可大幅提升推理速度 (141GB显存建议 False)

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==================== 内存管理辅助函数 ====================
def clear_memory():
    """强制清理 GPU 和 CPU 内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # 尝试释放 C 库的内存缓存 (Linux)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass


def get_memory_info():
    """获取当前内存使用情况"""
    info = []
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        info.append(f"GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.append(f"CPU: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB ({mem.percent}%)")
    except:
        pass
    return " | ".join(info)


def extract_keyframes_simple(video_path: str, output_dir: Path):
    """简化的关键帧提取 - 只保存首尾帧，避免加载额外模型"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"无法打开视频: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return
        
        # 读取第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if ret:
            cv2.imwrite(str(output_dir / f"{Path(video_path).stem}_keyframe0.jpg"), first_frame)
        
        # 读取中间帧
        if total_frames > 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, mid_frame = cap.read()
            if ret:
                cv2.imwrite(str(output_dir / f"{Path(video_path).stem}_keyframe1.jpg"), mid_frame)
        
        # 读取最后一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if ret:
            cv2.imwrite(str(output_dir / "last_frame.jpg"), last_frame)
            # 也保存最后 5 帧作为 motion_frames
            frames_for_motion = []
            for i in range(max(0, total_frames - 5), total_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames_for_motion.append(frame)
            
            if frames_for_motion:
                h, w = frames_for_motion[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                motion_path = str(output_dir / "motion_frames.mp4")
                writer = cv2.VideoWriter(motion_path, fourcc, 5, (w, h))
                for frame in frames_for_motion:
                    writer.write(frame)
                writer.release()
        
        cap.release()
        logger.info(f"简化关键帧提取完成: {video_path}")
    except Exception as e:
        logger.warning(f"关键帧提取失败: {e}")


# ==================== 全局模型实例 ====================
class ModelManager:
    """模型管理器，负责模型的加载和管理 - 同一时间只保留一个模型在内存中"""
    
    def __init__(self):
        self.t2v_model = None
        self.m2v_model_mi2v = None
        self.m2v_model_mm2v = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.t2v_config = None
        self.m2v_config = None
        self.current_model = None  # 记录当前加载的模型类型
    
    def _unload_other_models(self, keep: str = None):
        """卸载其他模型，只保留指定的模型"""
        logger.info(f"正在卸载其他模型 (保留: {keep})...")
        logger.info(f"卸载前内存状态: {get_memory_info()}")
        
        if keep != "t2v" and self.t2v_model is not None:
            logger.info("卸载 T2V 模型...")
            # 先将模型移到 CPU 并删除
            try:
                if hasattr(self.t2v_model, 'text_encoder'):
                    del self.t2v_model.text_encoder
                if hasattr(self.t2v_model, 'vae'):
                    del self.t2v_model.vae
                if hasattr(self.t2v_model, 'low_noise_model'):
                    del self.t2v_model.low_noise_model
                if hasattr(self.t2v_model, 'high_noise_model'):
                    del self.t2v_model.high_noise_model
            except:
                pass
            del self.t2v_model
            self.t2v_model = None
            gc.collect()
            torch.cuda.empty_cache()
        
        if keep != "mi2v" and self.m2v_model_mi2v is not None:
            logger.info("卸载 M2V (MI2V) 模型...")
            try:
                if hasattr(self.m2v_model_mi2v, 'text_encoder'):
                    del self.m2v_model_mi2v.text_encoder
                if hasattr(self.m2v_model_mi2v, 'vae'):
                    del self.m2v_model_mi2v.vae
                if hasattr(self.m2v_model_mi2v, 'low_noise_model'):
                    del self.m2v_model_mi2v.low_noise_model
                if hasattr(self.m2v_model_mi2v, 'high_noise_model'):
                    del self.m2v_model_mi2v.high_noise_model
            except:
                pass
            del self.m2v_model_mi2v
            self.m2v_model_mi2v = None
            gc.collect()
            torch.cuda.empty_cache()
        
        if keep != "mm2v" and self.m2v_model_mm2v is not None:
            logger.info("卸载 M2V (MM2V) 模型...")
            try:
                if hasattr(self.m2v_model_mm2v, 'text_encoder'):
                    del self.m2v_model_mm2v.text_encoder
                if hasattr(self.m2v_model_mm2v, 'vae'):
                    del self.m2v_model_mm2v.vae
                if hasattr(self.m2v_model_mm2v, 'low_noise_model'):
                    del self.m2v_model_mm2v.low_noise_model
                if hasattr(self.m2v_model_mm2v, 'high_noise_model'):
                    del self.m2v_model_mm2v.high_noise_model
            except:
                pass
            del self.m2v_model_mm2v
            self.m2v_model_mm2v = None
            gc.collect()
            torch.cuda.empty_cache()
        
        # 强制多次垃圾回收
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        logger.info(f"模型卸载完成，内存状态: {get_memory_info()}")
        
    def load_t2v_model(self):
        """加载 T2V 模型"""
        if self.t2v_model is not None:
            logger.info("T2V 模型已加载")
            return
        
        # 先卸载其他模型
        self._unload_other_models(keep="t2v")
        
        logger.info("正在加载 T2V 模型...")
        self.t2v_config = WAN_CONFIGS["t2v-A14B"]
        
        self.t2v_model = wan.WanT2V(
            config=self.t2v_config,
            checkpoint_dir=str(T2V_MODEL_PATH),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )
        self.current_model = "t2v"
        logger.info("T2V 模型加载完成")
        
    def load_m2v_model(self, mode="mi2v"):
        """加载 M2V 模型
        
        Args:
            mode: "mi2v" 或 "mm2v"
        """
        if mode == "mi2v":
            if self.m2v_model_mi2v is not None:
                logger.info("M2V (MI2V) 模型已加载")
                return self.m2v_model_mi2v
            lora_path = LORA_MI2V_PATH
        else:
            if self.m2v_model_mm2v is not None:
                logger.info("M2V (MM2V) 模型已加载")
                return self.m2v_model_mm2v
            lora_path = LORA_MM2V_PATH
        
        # 先卸载其他模型（包括 T2V）
        self._unload_other_models(keep=mode)
        
        logger.info(f"正在加载 M2V ({mode.upper()}) 模型...")
        logger.info(f"内存状态: {get_memory_info()}")
        
        self.m2v_config = WAN_CONFIGS["m2v-A14B"]
        
        # 配置 LoRA 权重路径
        self.m2v_config.low_noise_lora.weight = os.path.join(str(lora_path), "backbone_low_noise.safetensors")
        self.m2v_config.high_noise_lora.weight = os.path.join(str(lora_path), "backbone_high_noise.safetensors")
        self.m2v_config.low_noise_lora.r = self.m2v_config.low_noise_lora.lora_alpha = DEFAULT_LORA_RANK
        self.m2v_config.high_noise_lora.r = self.m2v_config.high_noise_lora.lora_alpha = DEFAULT_LORA_RANK
        
        m2v_model = wan.WanM2V(
            config=self.m2v_config,
            checkpoint_dir=str(I2V_MODEL_PATH),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )
        
        if mode == "mi2v":
            self.m2v_model_mi2v = m2v_model
        else:
            self.m2v_model_mm2v = m2v_model
        
        self.current_model = mode
        logger.info(f"M2V ({mode.upper()}) 模型加载完成")
        logger.info(f"内存状态: {get_memory_info()}")
        return m2v_model
    
    def unload_all(self):
        """卸载所有模型"""
        self._unload_other_models(keep=None)
        self.current_model = None
        logger.info("所有模型已卸载")


# 全局模型管理器
model_manager = ModelManager()


# ==================== 辅助函数 ====================
def get_story_files() -> List[str]:
    """获取所有故事脚本文件"""
    story_files = list(STORY_DIR.glob("*.json"))
    return [f.stem for f in sorted(story_files)]


def load_story_script(story_name: str) -> Dict[str, Any]:
    """加载故事脚本"""
    story_path = STORY_DIR / f"{story_name}.json"
    with open(story_path, "r", encoding="utf-8") as f:
        return json5.load(f)


def get_story_preview(story_name: str) -> str:
    """获取故事预览信息"""
    if not story_name:
        return "请选择一个故事脚本"
    
    try:
        script = load_story_script(story_name)
        preview = f"## 📖 {script.get('story_name', story_name)}\n\n"
        preview += f"**故事概述:**\n{script.get('story_overview', '无')}\n\n"
        
        scenes = script.get("scenes", [])
        preview += f"**场景数量:** {len(scenes)}\n\n"
        
        total_shots = sum(len(scene.get("video_prompts", [])) for scene in scenes)
        preview += f"**总镜头数:** {total_shots}\n\n"
        
        preview += "---\n### 场景详情:\n"
        for scene in scenes:
            scene_num = scene.get("scene_num", "?")
            prompts = scene.get("video_prompts", [])
            cuts = scene.get("cut", [])
            preview += f"\n**场景 {scene_num}** ({len(prompts)} 个镜头):\n"
            for i, prompt in enumerate(prompts):
                cut_info = "🎬 新镜头" if (i < len(cuts) and cuts[i]) else "➡️ 连续"
                preview += f"- {cut_info}: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
        
        return preview
    except Exception as e:
        return f"加载故事脚本失败: {str(e)}"


def create_output_dir(story_name: str) -> Path:
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{story_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def concat_videos(output_dir: Path, output_name: str) -> Optional[str]:
    """合并视频"""
    videos = sorted(glob.glob(str(output_dir / "*.mp4")))
    if not videos:
        return None
    
    # 过滤掉已经合并的视频和motion_frames
    videos = [v for v in videos if not v.endswith(f"{output_name}.mp4") and "motion_frames" not in v]
    
    if not videos:
        return None
    
    list_path = output_dir / "concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for v in videos:
            f.write(f"file '{os.path.abspath(v)}'\n")
    
    out_path = output_dir / f"{output_name}.mp4"
    
    ret = subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", "-y", str(out_path)],
        capture_output=True
    )
    
    if ret.returncode != 0:
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-r", "30", "-y", str(out_path)
        ], check=True)
    
    return str(out_path)


# ==================== 核心生成函数 ====================
def generate_first_shot_t2v(
    prompt: str,
    output_dir: Path,
    size: str,
    seed: int,
    guide_scale: float,
    frame_num: int = DEFAULT_FRAME_NUM,
    progress_callback=None
) -> Optional[str]:
    """使用 T2V 模型生成首个镜头"""
    logger.info(f"使用 T2V 生成首个镜头: {prompt[:50]}...")
    logger.info(f"内存状态: {get_memory_info()}")
    
    # 生成前清理内存
    clear_memory()
    
    model_manager.load_t2v_model()
    t2v_model = model_manager.t2v_model
    t2v_config = model_manager.t2v_config
    
    logger.info(f"开始 T2V 推理，帧数: {frame_num}...")
    
    try:
        video = t2v_model.generate(
            prompt,
            size=SIZE_CONFIGS[size],
            frame_num=frame_num,  # 使用自定义帧数
            shift=t2v_config.sample_shift,
            sample_solver='unipc',
            sampling_steps=t2v_config.sample_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=USE_OFFLOAD_MODEL
        )
        logger.info(f"T2V 推理完成，视频 tensor shape: {video.shape}")
        logger.info(f"内存状态: {get_memory_info()}")
        
        output_path = output_dir / "01_01.mp4"
        logger.info(f"正在保存视频到: {output_path}")
        
        # 使用 torch.no_grad() 减少内存
        with torch.no_grad():
            save_video(
                tensor=video[None],
                save_file=str(output_path),
                fps=t2v_config.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
        logger.info("视频保存完成")
        
        # 先清理显存
        del video
        clear_memory()
        logger.info("显存已清理")
        
        # 提取关键帧 - 使用简化版本
        logger.info("正在提取关键帧...")
        extract_keyframes_simple(str(output_path), output_dir)
        logger.info("关键帧提取完成")
        
        logger.info(f"首个镜头生成完成: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"T2V 生成失败: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()
        raise


def generate_next_shots_m2v(
    story_script: Dict[str, Any],
    output_dir: Path,
    size: str,
    max_memory_size: int,
    seed: int,
    guide_scale: float,
    mode: str = "mi2v",
    fix_keyframes: int = 3,
    skip_first: bool = True,
    frame_num: int = DEFAULT_FRAME_NUM,
    progress_callback=None
) -> List[str]:
    """使用 M2V 模型生成后续镜头"""
    logger.info(f"使用 M2V ({mode.upper()}) 生成后续镜头...")
    logger.info(f"内存状态: {get_memory_info()}")
    
    # 生成前清理内存
    clear_memory()
    
    m2v_model = model_manager.load_m2v_model(mode)
    m2v_config = model_manager.m2v_config
    
    generated_videos = []
    
    for scene in story_script["scenes"]:
        scene_num = scene["scene_num"]
        
        for i, prompt in enumerate(scene["video_prompts"]):
            shot_num = i + 1
            
            # 如果是第一个镜头且 skip_first=True，跳过
            if skip_first and scene_num == 1 and shot_num == 1:
                continue
            
            logger.info(f"生成场景 {scene_num} / 镜头 {shot_num}: {prompt[:50]}...")
            logger.info(f"内存状态: {get_memory_info()}")
            
            if progress_callback:
                progress_callback(f"正在生成场景 {scene_num} / 镜头 {shot_num}...")
            
            # 生成前清理内存
            clear_memory()
            
            # 获取记忆库
            memory_bank = sorted(glob.glob(str(output_dir / "*keyframe*.jpg")))
            if len(memory_bank) > max_memory_size:
                memory_bank = memory_bank[:fix_keyframes] + memory_bank[-(max_memory_size - fix_keyframes):]
            
            # 确定首帧和运动帧文件
            is_cut = scene["cut"][i] if i < len(scene.get("cut", [])) else True
            
            if mode == "mi2v" and not is_cut:
                first_frame_file = str(output_dir / "last_frame.jpg")
                if not os.path.exists(first_frame_file):
                    first_frame_file = None
            else:
                first_frame_file = None
            
            if mode == "mm2v" and not is_cut:
                motion_frames_file = str(output_dir / "motion_frames.mp4")
                if not os.path.exists(motion_frames_file):
                    motion_frames_file = None
            else:
                motion_frames_file = None
            
            try:
                video = m2v_model.generate(
                    prompt,
                    memory_bank,
                    first_frame_file=first_frame_file,
                    motion_frames_file=motion_frames_file,
                    max_area=MAX_AREA_CONFIGS[size],
                    frame_num=frame_num,  # 使用自定义帧数
                    shift=m2v_config.sample_shift,
                    sample_solver='unipc',
                    sampling_steps=m2v_config.sample_steps,
                    guide_scale=guide_scale,
                    seed=seed + i,
                    offload_model=USE_OFFLOAD_MODEL
                )
                
                # 处理视频帧
                if first_frame_file is not None:
                    video = video[:, 1:]
                elif motion_frames_file is not None:
                    video = video[:, 5:]
                
                output_path = output_dir / f"{scene_num:02d}_{shot_num:02d}.mp4"
                
                with torch.no_grad():
                    save_video(
                        tensor=video[None],
                        save_file=str(output_path),
                        fps=m2v_config.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1)
                    )
                
                # 先清理显存
                del video
                clear_memory()
                
                # 提取关键帧 - 使用简化版本
                extract_keyframes_simple(str(output_path), output_dir)
                
                generated_videos.append(str(output_path))
                logger.info(f"镜头生成完成: {output_path}")
                
            except Exception as e:
                logger.error(f"镜头生成失败: {e}")
                import traceback
                traceback.print_exc()
                clear_memory()
                continue
    
    return generated_videos


# ==================== Gradio 界面函数 ====================
def generate_story_video(
    story_name: str,
    size: str,
    max_memory_size: int,
    seed: int,
    guide_scale: float,
    mode: str,
    use_t2v_first: bool,
    frame_num: int = DEFAULT_FRAME_NUM,
    progress=gr.Progress()
):
    """生成完整故事视频的主函数"""
    if not story_name:
        return None, None, "❌ 请选择一个故事脚本"
    
    try:
        progress(0, desc="正在准备...")
        
        # 加载故事脚本
        story_script = load_story_script(story_name)
        output_dir = create_output_dir(story_name)
        
        log_messages = [f"🚀 开始生成故事视频: {story_script.get('story_name', story_name)}"]
        log_messages.append(f"📁 输出目录: {output_dir}")
        log_messages.append(f"🎞️ 帧数设置: {frame_num} (默认 81，减少帧数可节省内存)")
        
        total_scenes = len(story_script.get("scenes", []))
        total_shots = sum(len(scene.get("video_prompts", [])) for scene in story_script.get("scenes", []))
        
        log_messages.append(f"📊 总场景数: {total_scenes}, 总镜头数: {total_shots}")
        
        current_shot = 0
        
        # 生成首个镜头
        if use_t2v_first:
            progress(0.1, desc="正在使用 T2V 生成首个镜头...")
            first_prompt = story_script["scenes"][0]["video_prompts"][0]
            log_messages.append(f"\n🎬 使用 T2V 生成首个镜头...")
            log_messages.append(f"   Prompt: {first_prompt[:100]}...")
            
            first_video = generate_first_shot_t2v(
                prompt=first_prompt,
                output_dir=output_dir,
                size=size,
                seed=seed,
                guide_scale=guide_scale,
                frame_num=frame_num
            )
            
            if first_video:
                log_messages.append(f"✅ 首个镜头生成完成: {Path(first_video).name}")
            current_shot = 1
        
        # 生成后续镜头
        progress(0.2, desc=f"正在使用 M2V ({mode.upper()}) 生成后续镜头...")
        log_messages.append(f"\n🎬 使用 M2V ({mode.upper()}) 生成后续镜头...")
        
        def progress_callback(msg):
            progress(0.2 + 0.7 * (current_shot / total_shots), desc=msg)
        
        generated_videos = generate_next_shots_m2v(
            story_script=story_script,
            output_dir=output_dir,
            size=size,
            max_memory_size=max_memory_size,
            seed=seed,
            guide_scale=guide_scale,
            mode=mode,
            skip_first=use_t2v_first,
            frame_num=frame_num,
            progress_callback=progress_callback
        )
        
        for v in generated_videos:
            log_messages.append(f"✅ 镜头生成完成: {Path(v).name}")
        
        # 合并视频
        progress(0.95, desc="正在合并视频...")
        log_messages.append(f"\n🔗 正在合并所有视频...")
        
        final_video = concat_videos(output_dir, story_name)
        
        if final_video:
            log_messages.append(f"✅ 最终视频: {final_video}")
        
        progress(1.0, desc="完成!")
        log_messages.append(f"\n🎉 故事视频生成完成!")
        
        # 获取所有生成的视频
        all_videos = sorted(glob.glob(str(output_dir / "*.mp4")))
        video_gallery = [v for v in all_videos if "motion_frames" not in v]
        
        return final_video, video_gallery, "\n".join(log_messages)
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, None, error_msg


def generate_single_shot(
    prompt: str,
    memory_images: List[str],
    first_frame: Optional[str],
    size: str,
    seed: int,
    guide_scale: float,
    mode: str,
    frame_num: int = DEFAULT_FRAME_NUM,
    progress=gr.Progress()
):
    """生成单个镜头"""
    if not prompt:
        return None, "❌ 请输入 Prompt"
    
    try:
        progress(0.1, desc="正在准备...")
        
        # 清理内存
        clear_memory()
        
        # 创建临时输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"single_shot_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_messages = [f"🚀 开始生成单镜头视频"]
        log_messages.append(f"📁 输出目录: {output_dir}")
        log_messages.append(f"📝 Prompt: {prompt[:100]}...")
        log_messages.append(f"🎞️ 帧数: {frame_num}")
        
        # 处理记忆图片
        memory_bank = []
        if memory_images:
            for i, img_path in enumerate(memory_images):
                if img_path:
                    # 复制图片到输出目录
                    dst_path = output_dir / f"00_00_keyframe{i}.jpg"
                    shutil.copy(img_path, dst_path)
                    memory_bank.append(str(dst_path))
            log_messages.append(f"📷 记忆图片数量: {len(memory_bank)}")
        
        # 处理首帧
        first_frame_file = None
        if first_frame and mode == "mi2v":
            first_frame_file = str(output_dir / "last_frame.jpg")
            shutil.copy(first_frame, first_frame_file)
            log_messages.append(f"🖼️ 使用首帧: {first_frame}")
        
        progress(0.2, desc=f"正在使用 M2V ({mode.upper()}) 生成...")
        
        m2v_model = model_manager.load_m2v_model(mode)
        m2v_config = model_manager.m2v_config
        
        video = m2v_model.generate(
            prompt,
            memory_bank,
            first_frame_file=first_frame_file,
            motion_frames_file=None,
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,  # 使用自定义帧数
            shift=m2v_config.sample_shift,
            sample_solver='unipc',
            sampling_steps=m2v_config.sample_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=USE_OFFLOAD_MODEL
        )
        
        if first_frame_file is not None:
            video = video[:, 1:]
        
        output_path = output_dir / "output.mp4"
        
        with torch.no_grad():
            save_video(
                tensor=video[None],
                save_file=str(output_path),
                fps=m2v_config.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
        
        del video
        clear_memory()
        
        progress(1.0, desc="完成!")
        log_messages.append(f"\n✅ 视频生成完成: {output_path}")
        
        return str(output_path), "\n".join(log_messages)
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg


def generate_t2v_single(
    prompt: str,
    size: str,
    seed: int,
    guide_scale: float,
    frame_num: int = DEFAULT_FRAME_NUM,
    progress=gr.Progress()
):
    """使用 T2V 生成单个视频"""
    if not prompt:
        return None, "❌ 请输入 Prompt"
    
    try:
        progress(0.1, desc="正在准备...")
        
        # 创建临时输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"t2v_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_messages = [f"🚀 开始使用 T2V 生成视频"]
        log_messages.append(f"📁 输出目录: {output_dir}")
        log_messages.append(f"📝 Prompt: {prompt[:100]}...")
        log_messages.append(f"🎞️ 帧数: {frame_num}")
        
        progress(0.2, desc="正在使用 T2V 生成...")
        
        output_path = generate_first_shot_t2v(
            prompt=prompt,
            output_dir=output_dir,
            size=size,
            seed=seed,
            guide_scale=guide_scale,
            frame_num=frame_num
        )
        
        progress(1.0, desc="完成!")
        log_messages.append(f"\n✅ 视频生成完成: {output_path}")
        
        return output_path, "\n".join(log_messages)
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg


# ==================== 创建 Gradio 界面 ====================
def create_ui():
    """创建 Gradio 界面"""
    
    # 获取故事列表
    story_list = get_story_files()
    
    # 自定义 CSS
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .story-preview {
        max-height: 400px;
        overflow-y: auto;
    }
    """
    
    with gr.Blocks(
        title="🎬 StoryMem - 故事视频生成器",
        css=custom_css,
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🎬 StoryMem - 故事视频生成器
        
        基于 StoryMem 的故事视频生成工具，支持文本到视频 (T2V) 和记忆增强视频生成 (M2V)。
        
        **功能特点:**
        - 📖 完整故事生成：加载故事脚本，自动生成多镜头视频
        - 🎯 单镜头生成：灵活生成单个视频片段
        - 🖼️ 记忆增强：使用参考图片保持角色一致性
        - 🔗 场景连接：MI2V/MM2V 模式实现相邻镜头的平滑过渡
        """)
        
        with gr.Tabs():
            # ==================== Tab 1: 完整故事生成 ====================
            with gr.TabItem("📖 完整故事生成", id="story"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 生成设置")
                        
                        story_dropdown = gr.Dropdown(
                            label="选择故事脚本",
                            choices=story_list,
                            value=story_list[0] if story_list else None,
                            interactive=True
                        )
                        
                        refresh_btn = gr.Button("🔄 刷新故事列表", size="sm")
                        
                        with gr.Group():
                            size_dropdown = gr.Dropdown(
                                label="视频分辨率",
                                choices=list(SIZE_CONFIGS.keys()),
                                value=DEFAULT_SIZE
                            )
                            
                            frame_num_slider = gr.Slider(
                                label="视频帧数",
                                minimum=17,
                                maximum=81,
                                value=DEFAULT_FRAME_NUM,
                                step=4,
                                info="帧数越少内存占用越小 (17≈1秒, 41≈2.5秒, 81≈5秒)"
                            )
                            
                            max_memory = gr.Slider(
                                label="最大记忆帧数",
                                minimum=1,
                                maximum=20,
                                value=DEFAULT_MAX_MEMORY_SIZE,
                                step=1
                            )
                            
                            seed_input = gr.Number(
                                label="随机种子 (0=固定, -1=随机)",
                                value=DEFAULT_SEED,
                                precision=0
                            )
                            
                            guide_scale = gr.Slider(
                                label="引导强度 (Guidance Scale)",
                                minimum=1.0,
                                maximum=10.0,
                                value=DEFAULT_SAMPLE_GUIDE_SCALE,
                                step=0.5
                            )
                        
                        with gr.Group():
                            mode_radio = gr.Radio(
                                label="M2V 模式",
                                choices=["mi2v", "mm2v"],
                                value="mi2v",
                                info="MI2V: 首帧条件连接 | MM2V: 运动帧条件连接"
                            )
                            
                            t2v_first = gr.Checkbox(
                                label="使用 T2V 生成首个镜头",
                                value=True,
                                info="勾选后将使用 T2V 模型生成第一个镜头作为初始记忆"
                            )
                        
                        generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 故事预览")
                        story_preview = gr.Markdown(
                            value=get_story_preview(story_list[0]) if story_list else "无故事脚本",
                            elem_classes=["story-preview"]
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎬 生成结果")
                        output_video = gr.Video(label="最终视频")
                        
                    with gr.Column():
                        gr.Markdown("### 📹 所有镜头")
                        output_gallery = gr.Gallery(
                            label="生成的视频片段",
                            columns=3,
                            height="auto"
                        )
                
                output_log = gr.Textbox(
                    label="📝 生成日志",
                    lines=10,
                    max_lines=20
                )
                
                # 事件绑定
                story_dropdown.change(
                    fn=get_story_preview,
                    inputs=[story_dropdown],
                    outputs=[story_preview]
                )
                
                refresh_btn.click(
                    fn=lambda: gr.update(choices=get_story_files()),
                    outputs=[story_dropdown]
                )
                
                generate_btn.click(
                    fn=generate_story_video,
                    inputs=[
                        story_dropdown,
                        size_dropdown,
                        max_memory,
                        seed_input,
                        guide_scale,
                        mode_radio,
                        t2v_first,
                        frame_num_slider
                    ],
                    outputs=[output_video, output_gallery, output_log]
                )
            
            # ==================== Tab 2: 单镜头生成 (M2V) ====================
            with gr.TabItem("🎯 单镜头生成 (M2V)", id="single"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 生成设置")
                        
                        single_prompt = gr.Textbox(
                            label="视频描述 (Prompt)",
                            placeholder="Enter your video prompt here...",
                            lines=4
                        )
                        
                        gr.Markdown("### 🖼️ 记忆图片 (可选)")
                        memory_images = gr.File(
                            label="上传参考图片 (可多选)",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        first_frame_input = gr.Image(
                            label="首帧图片 (MI2V 模式可用)",
                            type="filepath"
                        )
                        
                        with gr.Group():
                            single_size = gr.Dropdown(
                                label="视频分辨率",
                                choices=list(SIZE_CONFIGS.keys()),
                                value=DEFAULT_SIZE
                            )
                            
                            single_frame_num = gr.Slider(
                                label="视频帧数",
                                minimum=17,
                                maximum=81,
                                value=DEFAULT_FRAME_NUM,
                                step=4,
                                info="帧数越少内存占用越小 (17≈1秒, 41≈2.5秒, 81≈5秒)"
                            )
                            
                            single_seed = gr.Number(
                                label="随机种子",
                                value=DEFAULT_SEED,
                                precision=0
                            )
                            
                            single_guide = gr.Slider(
                                label="引导强度",
                                minimum=1.0,
                                maximum=10.0,
                                value=DEFAULT_SAMPLE_GUIDE_SCALE,
                                step=0.5
                            )
                            
                            single_mode = gr.Radio(
                                label="M2V 模式",
                                choices=["mi2v", "mm2v"],
                                value="mi2v"
                            )
                        
                        single_generate_btn = gr.Button("🚀 生成视频", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎬 生成结果")
                        single_output_video = gr.Video(label="生成的视频")
                        single_output_log = gr.Textbox(
                            label="📝 生成日志",
                            lines=10
                        )
                
                # 事件绑定
                single_generate_btn.click(
                    fn=generate_single_shot,
                    inputs=[
                        single_prompt,
                        memory_images,
                        first_frame_input,
                        single_size,
                        single_seed,
                        single_guide,
                        single_mode,
                        single_frame_num
                    ],
                    outputs=[single_output_video, single_output_log]
                )
            
            # ==================== Tab 3: 文本到视频 (T2V) ====================
            with gr.TabItem("📝 文本到视频 (T2V)", id="t2v"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 生成设置")
                        
                        t2v_prompt = gr.Textbox(
                            label="视频描述 (Prompt)",
                            placeholder="Enter your video prompt here...",
                            lines=4
                        )
                        
                        with gr.Group():
                            t2v_size = gr.Dropdown(
                                label="视频分辨率",
                                choices=list(SIZE_CONFIGS.keys()),
                                value=DEFAULT_SIZE
                            )
                            
                            t2v_frame_num = gr.Slider(
                                label="视频帧数",
                                minimum=17,
                                maximum=81,
                                value=DEFAULT_FRAME_NUM,
                                step=4,
                                info="帧数越少内存占用越小 (17≈1秒, 41≈2.5秒, 81≈5秒)"
                            )
                            
                            t2v_seed = gr.Number(
                                label="随机种子",
                                value=DEFAULT_SEED,
                                precision=0
                            )
                            
                            t2v_guide = gr.Slider(
                                label="引导强度",
                                minimum=1.0,
                                maximum=10.0,
                                value=DEFAULT_SAMPLE_GUIDE_SCALE,
                                step=0.5
                            )
                        
                        t2v_generate_btn = gr.Button("🚀 生成视频", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎬 生成结果")
                        t2v_output_video = gr.Video(label="生成的视频")
                        t2v_output_log = gr.Textbox(
                            label="📝 生成日志",
                            lines=10
                        )
                
                # 示例 Prompts
                gr.Markdown("### 💡 示例 Prompts")
                gr.Examples(
                    examples=[
                        ["A cat walking on the beach at sunset, cinematic lighting, 4K quality"],
                        ["An astronaut floating in space with Earth in the background, realistic"],
                        ["A beautiful forest with sunlight streaming through the trees, peaceful atmosphere"],
                        ["A futuristic city at night with neon lights, cyberpunk style"],
                    ],
                    inputs=[t2v_prompt]
                )
                
                # 事件绑定
                t2v_generate_btn.click(
                    fn=generate_t2v_single,
                    inputs=[
                        t2v_prompt,
                        t2v_size,
                        t2v_seed,
                        t2v_guide,
                        t2v_frame_num
                    ],
                    outputs=[t2v_output_video, t2v_output_log]
                )
            
            # ==================== Tab 4: 故事脚本编辑器 ====================
            with gr.TabItem("✏️ 故事脚本编辑器", id="editor"):
                gr.Markdown("""
                ### 📝 故事脚本编辑器
                
                在这里可以创建或编辑故事脚本。脚本使用 JSON 格式。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        editor_story = gr.Dropdown(
                            label="选择要编辑的故事",
                            choices=["[新建故事]"] + story_list,
                            value="[新建故事]"
                        )
                        
                        story_name_input = gr.Textbox(
                            label="故事名称 (文件名)",
                            placeholder="my_story"
                        )
                        
                        load_btn = gr.Button("📂 加载故事", size="sm")
                        save_btn = gr.Button("💾 保存故事", variant="primary")
                    
                    with gr.Column(scale=2):
                        script_editor = gr.Code(
                            label="故事脚本 (JSON)",
                            language="json",
                            lines=30,
                            value='''{
  "story_name": "My Story",
  "story_overview": "A brief description of your story...",
  "scenes": [
    {
      "scene_num": 1,
      "video_prompts": [
        "First shot description...",
        "Second shot description..."
      ],
      "cut": [true, false]
    }
  ]
}'''
                        )
                
                editor_output = gr.Textbox(label="操作结果", lines=3)
                
                def load_story_for_edit(story_name):
                    if story_name == "[新建故事]":
                        return '''{
  "story_name": "My Story",
  "story_overview": "A brief description of your story...",
  "scenes": [
    {
      "scene_num": 1,
      "video_prompts": [
        "First shot description...",
        "Second shot description..."
      ],
      "cut": [true, false]
    }
  ]
}''', ""
                    try:
                        story_path = STORY_DIR / f"{story_name}.json"
                        with open(story_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        return content, f"✅ 已加载: {story_name}.json"
                    except Exception as e:
                        return "", f"❌ 加载失败: {str(e)}"
                
                def save_story_script(name, content):
                    if not name:
                        return "❌ 请输入故事名称"
                    try:
                        # 验证 JSON 格式
                        json5.loads(content)
                        
                        story_path = STORY_DIR / f"{name}.json"
                        with open(story_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        return f"✅ 已保存: {story_path}"
                    except json5.JSON5DecodeError as e:
                        return f"❌ JSON 格式错误: {str(e)}"
                    except Exception as e:
                        return f"❌ 保存失败: {str(e)}"
                
                load_btn.click(
                    fn=load_story_for_edit,
                    inputs=[editor_story],
                    outputs=[script_editor, editor_output]
                )
                
                save_btn.click(
                    fn=save_story_script,
                    inputs=[story_name_input, script_editor],
                    outputs=[editor_output]
                )
        
        # 底部信息
        gr.Markdown("""
        ---
        ### 📌 使用说明
        
        1. **完整故事生成**: 选择预设的故事脚本，自动生成完整的多镜头视频
        2. **单镜头生成 (M2V)**: 使用记忆图片生成单个视频片段，保持角色一致性
        3. **文本到视频 (T2V)**: 直接从文本描述生成视频
        4. **故事脚本编辑器**: 创建或编辑自定义故事脚本
        
        **关于 M2V 模式:**
        - **MI2V**: Memory + First-frame Image - 使用上一镜头的最后一帧作为新镜头的首帧条件
        - **MM2V**: Memory + Motion Frames - 使用上一镜头的最后5帧作为运动条件
        
        **提示:** 
        - 首次运行需要加载模型，请耐心等待
        - 建议使用 832*480 分辨率以获得更好的效果
        - 设置随机种子为 0 可以获得可复现的结果
        """)
    
    return demo


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 打印配置信息
    logger.info("=" * 50)
    logger.info("StoryMem Gradio Web 界面")
    logger.info("=" * 50)
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"T2V 模型路径: {T2V_MODEL_PATH}")
    logger.info(f"I2V 模型路径: {I2V_MODEL_PATH}")
    logger.info(f"MI2V LoRA 路径: {LORA_MI2V_PATH}")
    logger.info(f"MM2V LoRA 路径: {LORA_MM2V_PATH}")
    logger.info(f"故事脚本目录: {STORY_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"默认帧数: {DEFAULT_FRAME_NUM} (原始 81 帧，减少帧数以节省内存)")
    logger.info(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info("=" * 50)
    
    # 不再预加载模型，在使用时加载以节省内存
    logger.info("⚠️ 模型将在首次使用时加载，请耐心等待...")
    
    # 创建并启动界面
    demo = create_ui()
    demo.queue(max_size=5)  # 减少队列大小
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
