#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SoulX-FlashTalk Gradio Web 应用
基于官方推理脚本改编，实现 Web 化演示
"""

import os
import sys
import time
import tempfile
import subprocess
from datetime import datetime
from collections import deque

import numpy as np
import torch
import librosa
import imageio
import gradio as gr
from loguru import logger

# 导入 FlashTalk 推理模块
from flash_talk.inference import (
    get_pipeline, 
    get_base_data, 
    get_audio_embedding, 
    run_pipeline, 
    infer_params
)

# ==================== 全局配置 ====================
CKPT_DIR = "checkpoints/SoulX-FlashTalk-14B"
WAV2VEC_DIR = "checkpoints/chinese-wav2vec2-base"
CPU_OFFLOAD = False  # H20 有 141GB 显存，无需 CPU offload

# 推理参数
SAMPLE_RATE = infer_params['sample_rate']
TGT_FPS = infer_params['tgt_fps']
CACHED_AUDIO_DURATION = infer_params['cached_audio_duration']
FRAME_NUM = infer_params['frame_num']
MOTION_FRAMES_NUM = infer_params['motion_frames_num']
SLICE_LEN = FRAME_NUM - MOTION_FRAMES_NUM

# 全局 Pipeline（启动时加载）
pipeline = None

# ==================== 示例数据 ====================
# 每个示例包含：(图片路径, 音频路径, 提示词)
EXAMPLES = [
    {
        "name": "示例 1",
        "image": "examples/man.png",
        "audio": "examples/cantonese_16k.wav",
        "prompt": "A person is talking. Only the foreground characters are moving, the background remains static."
    },
    {
        "name": "示例 2",
        "image": "examples/man.png",
        "audio": "examples/cantonese_16k.wav",
        "prompt": "A young woman is speaking passionately. Only the foreground characters are moving, the background remains static."
    },
    {
        "name": "示例 3",
        "image": "examples/man.png",
        "audio": "examples/cantonese_16k.wav",
        "prompt": "A man in a suit is giving a speech. Only the foreground characters are moving, the background remains static."
    },
    {
        "name": "示例 4",
        "image": "examples/man.png",
        "audio": "examples/cantonese_16k.wav",
        "prompt": "An elderly person is telling a story with expressive gestures. Only the foreground characters are moving, the background remains static."
    },
]

# ==================== 模型加载 ====================
def load_model():
    """启动时加载模型"""
    global pipeline
    logger.info("正在加载 SoulX-FlashTalk 模型...")
    logger.info(f"模型路径: {CKPT_DIR}")
    logger.info(f"Wav2Vec 路径: {WAV2VEC_DIR}")
    
    start_time = time.time()
    
    # 单 GPU 模式
    world_size = 1
    pipeline = get_pipeline(
        world_size=world_size, 
        ckpt_dir=CKPT_DIR, 
        wav2vec_dir=WAV2VEC_DIR, 
        cpu_offload=CPU_OFFLOAD
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"模型加载完成！耗时: {elapsed_time:.2f} 秒")
    
    return pipeline

# ==================== 视频保存 ====================
def save_video(frames_list, video_path, audio_path, fps):
    """保存视频并合并音频"""
    temp_video_path = video_path.replace('.mp4', '_temp.mp4')
    
    with imageio.get_writer(
        temp_video_path, 
        format='mp4', 
        mode='I',
        fps=fps, 
        codec='h264', 
        ffmpeg_params=['-bf', '0']
    ) as writer:
        for frames in frames_list:
            frames_np = frames.numpy().astype(np.uint8)
            for i in range(frames_np.shape[0]):
                frame = frames_np[i, :, :, :]
                writer.append_data(frame)
    
    # 合并视频和音频
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_video_path, 
        '-i', audio_path, 
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest', 
        video_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    # 删除临时文件
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    
    return video_path

# ==================== 生成视频 ====================
def generate_video(
    input_image,
    audio_file,
    prompt,
    seed,
    audio_encode_mode,
    progress=gr.Progress()
):
    """生成说话视频"""
    global pipeline
    
    if pipeline is None:
        return None, "❌ 错误：模型未加载，请刷新页面重试"
    
    if input_image is None:
        return None, "❌ 错误：请上传一张人物图片"
    
    if audio_file is None:
        return None, "❌ 错误：请上传一段音频文件"
    
    try:
        progress(0.1, desc="正在准备数据...")
        
        # 保存上传的图片
        temp_image_path = tempfile.mktemp(suffix=".png")
        input_image.save(temp_image_path)
        
        # 准备基础数据
        base_seed = seed if seed >= 0 else 9999
        get_base_data(
            pipeline, 
            input_prompt=prompt, 
            cond_image=temp_image_path, 
            base_seed=base_seed
        )
        
        progress(0.2, desc="正在加载音频...")
        
        # 加载音频
        human_speech_array_all, _ = librosa.load(
            audio_file, 
            sr=SAMPLE_RATE, 
            mono=True
        )
        
        # 计算每个 slice 对应的音频采样点数
        human_speech_array_slice_len = SLICE_LEN * SAMPLE_RATE // TGT_FPS
        
        # 在音频末尾添加静音，确保最后一段音频不被截断
        remainder = len(human_speech_array_all) % human_speech_array_slice_len
        if remainder > 0:
            # 需要补齐的静音长度
            padding_len = human_speech_array_slice_len - remainder
            # 添加静音（零值）
            human_speech_array_all = np.concatenate([
                human_speech_array_all, 
                np.zeros(padding_len, dtype=human_speech_array_all.dtype)
            ])
            logger.info(f"音频末尾添加 {padding_len / SAMPLE_RATE:.2f} 秒静音以补齐")
        
        generated_list = []
        
        progress(0.3, desc="正在生成视频...")
        
        if audio_encode_mode == "once":
            # 一次性编码音频
            audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)
            num_chunks = (audio_embedding_all.shape[1] - FRAME_NUM) // SLICE_LEN
            audio_embedding_chunks_list = [
                audio_embedding_all[:, i * SLICE_LEN: i * SLICE_LEN + FRAME_NUM].contiguous() 
                for i in range(num_chunks)
            ]
            
            for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
                progress_val = 0.3 + 0.6 * (chunk_idx + 1) / len(audio_embedding_chunks_list)
                progress(progress_val, desc=f"正在生成第 {chunk_idx + 1}/{len(audio_embedding_chunks_list)} 段视频...")
                
                start_time = time.time()
                video = run_pipeline(pipeline, audio_embedding_chunk)
                elapsed = time.time() - start_time
                
                logger.info(f"生成第 {chunk_idx + 1} 段视频完成，耗时: {elapsed:.2f}s")
                generated_list.append(video.cpu())
                
        else:  # stream 模式
            cached_audio_length_sum = SAMPLE_RATE * CACHED_AUDIO_DURATION
            audio_end_idx = CACHED_AUDIO_DURATION * TGT_FPS
            audio_start_idx = audio_end_idx - FRAME_NUM
            
            audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
            
            # 使用前面已计算的 slice 长度
            num_slices = len(human_speech_array_all) // human_speech_array_slice_len
            human_speech_array_slices = human_speech_array_all[
                :num_slices * human_speech_array_slice_len
            ].reshape(-1, human_speech_array_slice_len)
            
            for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
                progress_val = 0.3 + 0.6 * (chunk_idx + 1) / len(human_speech_array_slices)
                progress(progress_val, desc=f"正在生成第 {chunk_idx + 1}/{len(human_speech_array_slices)} 段视频...")
                
                # 流式编码音频
                audio_dq.extend(human_speech_array.tolist())
                audio_array = np.array(audio_dq)
                audio_embedding = get_audio_embedding(
                    pipeline, audio_array, audio_start_idx, audio_end_idx
                )
                
                start_time = time.time()
                video = run_pipeline(pipeline, audio_embedding)
                elapsed = time.time() - start_time
                
                logger.info(f"生成第 {chunk_idx + 1} 段视频完成，耗时: {elapsed:.2f}s")
                generated_list.append(video.cpu())
        
        progress(0.95, desc="正在保存视频...")
        
        # 保存视频
        output_dir = "sample_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"result_{timestamp}.mp4")
        
        save_video(generated_list, output_path, audio_file, fps=TGT_FPS)
        
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        progress(1.0, desc="完成!")
        
        return output_path, f"✅ 视频生成成功！保存至: {output_path}"
        
    except Exception as e:
        logger.error(f"生成视频时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"❌ 生成失败: {str(e)}"

# ==================== Gradio 界面 ====================
def create_ui():
    """创建 Gradio Web 界面"""
    
    # 自定义 CSS
    custom_css = """
    .youtube-banner {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .youtube-banner a {
        color: white !important;
        font-size: 18px;
        font-weight: bold;
        text-decoration: none;
    }
    .youtube-banner a:hover {
        text-decoration: underline;
    }
    .title-text {
        text-align: center;
        margin-bottom: 10px;
    }
    """
    
    with gr.Blocks(
        title="SoulX-FlashTalk - 实时音频驱动数字人",
        css=custom_css,
        theme=gr.themes.Soft()
    ) as demo:
        
        # YouTube 频道信息
        gr.HTML("""
        <div class="youtube-banner">
            <a href="https://www.youtube.com/@rongyi-ai" target="_blank">
                📺 AI 技术分享频道 | YouTube: https://www.youtube.com/@rongyi-ai
            </a>
        </div>
        """)
        
        # 标题
        gr.Markdown("""
        # 🎬 SoulX-FlashTalk: 实时音频驱动数字人视频生成
        
        上传一张人物图片和一段音频，即可生成逼真的说话视频。
        
        ---
        """)
        
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入设置")
                
                input_image = gr.Image(
                    label="人物图片",
                    type="pil",
                    height=300,
                )
                gr.Markdown("<small>💡 提示：建议上传竖版人物照片（比例约 9:16），模型输出固定为 448×768</small>")
                
                audio_file = gr.Audio(
                    label="音频文件",
                    type="filepath",
                )
                
                prompt = gr.Textbox(
                    label="提示词",
                    placeholder="请输入描述视频的提示词...",
                    value=EXAMPLES[0]["prompt"],
                    lines=3,
                )
                
                gr.Markdown("#### 示例（点击加载图片、音频和提示词）")
                example_btns = []
                for i, example in enumerate(EXAMPLES):
                    btn = gr.Button(
                        example["name"],
                        size="sm",
                        variant="secondary"
                    )
                    example_btns.append((btn, example))
                
                with gr.Accordion("⚙️ 高级设置", open=False):
                    seed = gr.Slider(
                        label="随机种子",
                        minimum=-1,
                        maximum=99999,
                        value=9999,
                        step=1,
                        info="-1 表示使用默认种子 (9999)"
                    )
                    
                    audio_encode_mode = gr.Radio(
                        label="音频编码模式",
                        choices=["stream", "once"],
                        value="stream",
                        info="stream: 流式编码（推荐）；once: 一次性编码"
                    )
                
                generate_btn = gr.Button(
                    "🚀 开始生成",
                    variant="primary",
                    size="lg"
                )
            
            # 右侧：输出区域
            with gr.Column(scale=1):
                gr.Markdown("### 📤 生成结果")
                
                output_video = gr.Video(
                    label="生成的视频",
                    height=400,
                )
                
                status_text = gr.Textbox(
                    label="状态信息",
                    interactive=False,
                    lines=2,
                )
        
        # 使用说明
        gr.Markdown("""
        ---
        ### 📖 使用说明
        
        1. **上传图片**：选择一张正面人物照片（建议清晰、光线均匀）
        2. **上传音频**：选择一段音频文件（支持 WAV、MP3 等格式，推荐 16kHz 采样率）
        3. **输入提示词**：描述视频中人物的状态（可使用示例提示词）
        4. **点击生成**：等待视频生成完成
        
        ### ⚠️ 注意事项
        
        - 首次生成可能需要较长时间进行模型预热
        - 生成时间与音频长度相关
        - 建议使用高质量的人物正面照片
        
        ---
        <div style="text-align: center; color: #666;">
            基于 <a href="https://github.com/Soul-AILab/SoulX-FlashTalk" target="_blank">SoulX-FlashTalk</a> 开源项目 | 
            模型由 Soul AI Lab 提供
        </div>
        """)
        
        # 加载示例的函数
        def load_example(example):
            from PIL import Image
            img = Image.open(example["image"])
            return img, example["audio"], example["prompt"]
        
        # 绑定示例按钮事件
        for btn, example in example_btns:
            btn.click(
                fn=lambda e=example: load_example(e),
                inputs=[],
                outputs=[input_image, audio_file, prompt]
            )
        
        # 绑定生成按钮事件
        generate_btn.click(
            fn=generate_video,
            inputs=[
                input_image,
                audio_file,
                prompt,
                seed,
                audio_encode_mode,
            ],
            outputs=[output_video, status_text]
        )
        
        # 添加示例
        gr.Examples(
            examples=[
                [
                    "examples/man.png",
                    "examples/cantonese_16k.wav",
                    "A person is talking. Only the foreground characters are moving, the background remains static.",
                    9999,
                    "stream"
                ],
            ],
            inputs=[
                input_image,
                audio_file,
                prompt,
                seed,
                audio_encode_mode,
            ],
            outputs=[output_video, status_text],
            fn=generate_video,
            cache_examples=False,
            label="📌 快速示例"
        )
    
    return demo


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 启动时加载模型
    logger.info("=" * 50)
    logger.info("SoulX-FlashTalk Web 应用启动中...")
    logger.info("=" * 50)
    
    # 加载模型
    load_model()
    
    # 创建并启动 Gradio 应用
    demo = create_ui()
    
    demo.queue(max_size=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
