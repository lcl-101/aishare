#!/usr/bin/env python3
"""
LTX-2 Gradio Web Application
支持3种核心视频生成功能：
1. 文本生成视频
2. 图片生成视频
3. 首尾帧插值
"""

import os
import logging
import gradio as gr
import torch
import gc

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 模型路径配置 ====================
CHECKPOINT_DIR = "checkpoints"
LTX2_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "LTX-2/ltx-2-19b-dev-fp8.safetensors")
SPATIAL_UPSAMPLER_PATH = os.path.join(CHECKPOINT_DIR, "LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors")
DISTILLED_LORA_PATH = os.path.join(CHECKPOINT_DIR, "LTX-2/ltx-2-19b-distilled-lora-384.safetensors")
GEMMA_ROOT = os.path.join(CHECKPOINT_DIR, "gemma-3-12b-it-qat-q4_0-unquantized")

# 全局管道对象
pipelines = {}


def get_device():
    """获取可用设备"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def cleanup_memory():
    """清理GPU显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def cleanup_all_pipelines():
    """清理所有管道"""
    global pipelines
    logger.info("正在清理所有管道...")
    for name in list(pipelines.keys()):
        pipeline = pipelines.pop(name)
        del pipeline
    
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    logger.info("所有管道已清理")


def get_pipeline(pipeline_name):
    """获取或初始化管道"""
    global pipelines
    
    # 如果已加载，直接返回
    if pipeline_name in pipelines:
        logger.info(f"使用已加载的 {pipeline_name} 管道")
        return pipelines[pipeline_name]
    
    # 清理其他管道
    if len(pipelines) > 0:
        logger.info(f"切换管道，清理已加载: {list(pipelines.keys())}")
        cleanup_all_pipelines()
    
    logger.info(f"正在初始化 {pipeline_name} 管道...")
    
    # 创建 LoRA 配置
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            path=DISTILLED_LORA_PATH,
            strength=1.0,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP
        )
    ]
    
    if pipeline_name == "ti2vid_two_stages":
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=LTX2_MODEL_PATH,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
            gemma_root=GEMMA_ROOT,
            loras=[],
            device=get_device(),
            fp8transformer=True,
        )
    elif pipeline_name == "keyframe_interpolation":
        pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=LTX2_MODEL_PATH,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
            gemma_root=GEMMA_ROOT,
            loras=[],
            device=get_device(),
            fp8transformer=True,
        )
    else:
        raise ValueError(f"未知的管道: {pipeline_name}")
    
    pipelines[pipeline_name] = pipeline
    logger.info(f"{pipeline_name} 管道初始化成功")
    return pipeline


# ==================== 生成函数 ====================

def generate_text_to_video(prompt, negative_prompt, seed, height, width, num_frames,
                           frame_rate, num_inference_steps, cfg_guidance_scale,
                           enhance_prompt, progress=gr.Progress()):
    """文本生成视频"""
    try:
        progress(0, desc="正在初始化管道...")
        pipeline = get_pipeline("ti2vid_two_stages")
        
        # 使用 tiling_config 支持大分辨率视频解码
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(int(num_frames), tiling_config)
        
        progress(0.1, desc="正在生成视频...")
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            num_inference_steps=int(num_inference_steps),
            cfg_guidance_scale=float(cfg_guidance_scale),
            images=[],  # 无图片输入
            enhance_prompt=enhance_prompt,
            tiling_config=tiling_config,
        )
        
        progress(0.9, desc="正在保存视频...")
        output_path = f"/tmp/ltx2_t2v_{int(seed)}.mp4"
        with torch.inference_mode():
            encode_video(
                video=video,
                fps=float(frame_rate),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )
        
        progress(1.0, desc="完成!")
        cleanup_memory()
        return output_path
    except Exception as e:
        logger.error(f"生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_image_to_video(input_image, image_strength, prompt, negative_prompt, seed, 
                            height, width, num_frames, frame_rate, num_inference_steps, 
                            cfg_guidance_scale, enhance_prompt, progress=gr.Progress()):
    """图片生成视频"""
    try:
        if input_image is None:
            raise ValueError("请上传一张图片！")
        
        progress(0, desc="正在初始化管道...")
        pipeline = get_pipeline("ti2vid_two_stages")
        
        # 使用 tiling_config 支持大分辨率视频解码
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(int(num_frames), tiling_config)
        
        # 构建图片条件：(图片路径, 帧索引, 强度)
        images = [(input_image, 0, float(image_strength))]
        
        progress(0.1, desc="正在生成视频...")
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            num_inference_steps=int(num_inference_steps),
            cfg_guidance_scale=float(cfg_guidance_scale),
            images=images,
            enhance_prompt=enhance_prompt,
            tiling_config=tiling_config,
        )
        
        progress(0.9, desc="正在保存视频...")
        output_path = f"/tmp/ltx2_i2v_{int(seed)}.mp4"
        with torch.inference_mode():
            encode_video(
                video=video,
                fps=float(frame_rate),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )
        
        progress(1.0, desc="完成!")
        cleanup_memory()
        return output_path
    except Exception as e:
        logger.error(f"生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_keyframe_interpolation(start_image, end_image, start_strength, end_strength,
                                    prompt, negative_prompt, seed, height, width, num_frames,
                                    frame_rate, num_inference_steps, cfg_guidance_scale,
                                    enhance_prompt, progress=gr.Progress()):
    """首尾帧插值"""
    try:
        if start_image is None or end_image is None:
            raise ValueError("请上传起始帧和结束帧图片！")
        
        progress(0, desc="正在初始化管道...")
        pipeline = get_pipeline("keyframe_interpolation")
        
        # 使用 tiling_config 支持大分辨率视频解码
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(int(num_frames), tiling_config)
        
        # 构建图片列表: (路径, 帧索引, 强度)
        images = [
            (start_image, 0, float(start_strength)),
            (end_image, int(num_frames) - 1, float(end_strength)),
        ]
        
        progress(0.1, desc="正在生成视频...")
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            num_inference_steps=int(num_inference_steps),
            cfg_guidance_scale=float(cfg_guidance_scale),
            images=images,
            enhance_prompt=enhance_prompt,
            tiling_config=tiling_config,
        )
        
        progress(0.9, desc="正在保存视频...")
        output_path = f"/tmp/ltx2_keyframe_{int(seed)}.mp4"
        with torch.inference_mode():
            encode_video(
                video=video,
                fps=float(frame_rate),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )
        
        progress(1.0, desc="完成!")
        cleanup_memory()
        return output_path
    except Exception as e:
        logger.error(f"生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== Gradio 界面 ====================

def create_demo():
    """创建Gradio界面"""
    with gr.Blocks(title="LTX-2 视频生成", theme=gr.themes.Soft()) as demo:
        # 顶部信息
        gr.Markdown("""
        # 🎬 LTX-2 视频生成平台
        ### 📺 [AI 技术分享频道](https://www.youtube.com/@rongyikanshijie-ai)
        欢迎订阅我的YouTube频道，获取更多AI技术教程和分享！
        
        ---
        **支持功能**: 文本生成视频 | 图片生成视频 | 首尾帧插值
        """)
        
        with gr.Tabs():
            # ==================== Tab 1: 文本生成视频 ====================
            with gr.Tab("📝 文本生成视频"):
                gr.Markdown("""
                ### 从文字描述生成视频和同步音频
                输入详细的场景描述，AI 会生成对应的视频内容。
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        t2v_prompt = gr.Textbox(
                            label="提示词 (Prompt)",
                            placeholder="描述你想要生成的视频内容...",
                            lines=4,
                            value="A serene lake surrounded by mountains at sunset, with reflections on the water and birds flying across the sky"
                        )
                        t2v_negative_prompt = gr.Textbox(
                            label="负面提示词",
                            placeholder="不想出现的内容...",
                            lines=2,
                            value="blurry, low quality, distorted, watermark"
                        )
                        t2v_enhance_prompt = gr.Checkbox(label="✨ 增强提示词 (AI优化)", value=False)
                        
                    with gr.Column(scale=1):
                        t2v_seed = gr.Number(label="随机种子", value=42, precision=0)
                        t2v_cfg_scale = gr.Slider(label="CFG引导强度", minimum=1.0, maximum=20.0, value=7.5, step=0.5)
                        t2v_steps = gr.Slider(label="推理步数", minimum=10, maximum=50, value=40, step=1)
                
                with gr.Row():
                    t2v_width = gr.Slider(label="输出宽度", minimum=512, maximum=1024, value=768, step=64,
                                         info="最终视频宽度(Stage1生成一半,Stage2上采样)")
                    t2v_height = gr.Slider(label="输出高度", minimum=512, maximum=1024, value=768, step=64,
                                          info="最终视频高度(Stage1生成一半,Stage2上采样)")
                    t2v_num_frames = gr.Slider(label="帧数 (1+8k)", minimum=17, maximum=257, value=65, step=8, 
                                               info="帧数必须是1+8k格式: 17,25,33...257")
                    t2v_fps = gr.Slider(label="帧率 (FPS)", minimum=8, maximum=30, value=24, step=1)
                
                t2v_generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        t2v_output = gr.Video(label="生成的视频", height=400)
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 📌 提示
                        - 点击右上角下载按钮获取原始分辨率视频
                        - 预览可能显得模糊，实际视频是高清的
                        - 建议分辨率：768×768 或 1024×1024
                        """)
                
                t2v_generate_btn.click(
                    fn=generate_text_to_video,
                    inputs=[t2v_prompt, t2v_negative_prompt, t2v_seed, t2v_height, t2v_width,
                           t2v_num_frames, t2v_fps, t2v_steps, t2v_cfg_scale, t2v_enhance_prompt],
                    outputs=t2v_output
                )
            
            # ==================== Tab 2: 图片生成视频 ====================
            with gr.Tab("🖼️ 图片生成视频"):
                gr.Markdown("""
                ### 从一张图片生成视频
                上传一张图片作为视频的第一帧，AI 会基于图片内容生成动态视频。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_image = gr.Image(
                            label="📷 上传图片 (作为第一帧)",
                            type="filepath",
                            height=300,
                        )
                        i2v_strength = gr.Slider(
                            label="图片强度", 
                            minimum=0.1, maximum=1.0, value=1.0, step=0.1,
                            info="控制图片对生成结果的影响程度"
                        )
                    
                    with gr.Column(scale=1):
                        i2v_prompt = gr.Textbox(
                            label="提示词 (描述动作/变化)",
                            placeholder="描述图片中的内容应该如何运动...",
                            lines=4,
                            value="A beautiful anime elf girl with long flowing silver-white hair and sparkling purple eyes. She looks at the camera, smiles warmly, and says \"Hello LTX 2\" with clear, gentle feminine voice. Her hair gently sways in a soft magical breeze. She blinks slowly and tilts her head. Soft wind chimes ring gently in the background, accompanied by ambient forest sounds. Dreamy purple gradient background with floating sparkles. Soft ethereal lighting, smooth high-quality anime animation, cinematic composition. Camera slowly pushes in."
                        )
                        i2v_negative_prompt = gr.Textbox(
                            label="负面提示词",
                            lines=2,
                            value="blurry, low quality, distorted, jerky motion, static, frozen"
                        )
                        i2v_enhance_prompt = gr.Checkbox(label="✨ 增强提示词", value=False)
                        i2v_seed = gr.Number(label="随机种子", value=123, precision=0)
                
                with gr.Row():
                    i2v_cfg_scale = gr.Slider(label="CFG引导强度", minimum=1.0, maximum=20.0, value=7.5, step=0.5)
                    i2v_steps = gr.Slider(label="推理步数", minimum=10, maximum=50, value=40, step=1)
                
                with gr.Row():
                    i2v_width = gr.Slider(label="输出宽度", minimum=512, maximum=1024, value=768, step=64,
                                         info="最终视频宽度")
                    i2v_height = gr.Slider(label="输出高度", minimum=512, maximum=1024, value=768, step=64,
                                          info="最终视频高度")
                    i2v_num_frames = gr.Slider(label="帧数 (1+8k)", minimum=17, maximum=257, value=65, step=8)
                    i2v_fps = gr.Slider(label="帧率 (FPS)", minimum=8, maximum=30, value=24, step=1)
                
                i2v_generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        i2v_output = gr.Video(label="生成的视频", height=400)
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 📌 提示
                        - 点击右上角下载按钮获取高清视频
                        - 图片会作为视频第一帧
                        - 提示词描述动作和变化
                        """)
                
                i2v_generate_btn.click(
                    fn=generate_image_to_video,
                    inputs=[i2v_image, i2v_strength, i2v_prompt, i2v_negative_prompt, i2v_seed,
                           i2v_height, i2v_width, i2v_num_frames, i2v_fps, i2v_steps,
                           i2v_cfg_scale, i2v_enhance_prompt],
                    outputs=i2v_output
                )
            
            # ==================== Tab 3: 首尾帧插值 ====================
            with gr.Tab("🎞️ 首尾帧插值"):
                gr.Markdown("""
                ### 在两张图片之间生成过渡动画
                上传起始帧和结束帧图片，AI 会生成它们之间的平滑过渡视频。
                """)
                
                with gr.Row():
                    with gr.Column():
                        kf_start_image = gr.Image(
                            label="🖼️ 起始帧图片",
                            type="filepath",
                            height=256,
                        )
                        kf_start_strength = gr.Slider(
                            label="起始帧强度", 
                            minimum=0.1, maximum=1.0, value=1.0, step=0.1
                        )
                    with gr.Column():
                        kf_end_image = gr.Image(
                            label="🖼️ 结束帧图片",
                            type="filepath",
                            height=256,
                        )
                        kf_end_strength = gr.Slider(
                            label="结束帧强度", 
                            minimum=0.1, maximum=1.0, value=1.0, step=0.1
                        )
                
                with gr.Row():
                    with gr.Column():
                        kf_prompt = gr.Textbox(
                            label="提示词 (描述过渡效果)",
                            placeholder="描述两张图片之间的过渡方式...",
                            lines=3,
                            value="Smooth cinematic transition with natural lighting changes"
                        )
                        kf_negative_prompt = gr.Textbox(
                            label="负面提示词",
                            lines=2,
                            value="blurry, low quality, jerky motion, flickering"
                        )
                        kf_enhance_prompt = gr.Checkbox(label="✨ 增强提示词", value=False)
                        
                    with gr.Column():
                        kf_seed = gr.Number(label="随机种子", value=789, precision=0)
                        kf_cfg_scale = gr.Slider(label="CFG引导强度", minimum=1.0, maximum=20.0, value=7.5, step=0.5)
                        kf_steps = gr.Slider(label="推理步数", minimum=10, maximum=50, value=40, step=1)
                
                with gr.Row():
                    kf_width = gr.Slider(label="输出宽度", minimum=512, maximum=1024, value=768, step=64,
                                        info="最终视频宽度")
                    kf_height = gr.Slider(label="输出高度", minimum=512, maximum=1024, value=768, step=64,
                                         info="最终视频高度")
                    kf_num_frames = gr.Slider(label="帧数 (1+8k)", minimum=17, maximum=257, value=65, step=8)
                    kf_fps = gr.Slider(label="帧率 (FPS)", minimum=8, maximum=30, value=24, step=1)
                
                kf_generate_btn = gr.Button("🎬 生成插值视频", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        kf_output = gr.Video(label="生成的视频", height=400)
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 📌 提示
                        - 点击右上角下载按钮获取高清视频
                        - 起始帧→结束帧的平滑过渡
                        - 提示词描述过渡效果
                        """)
                
                kf_generate_btn.click(
                    fn=generate_keyframe_interpolation,
                    inputs=[kf_start_image, kf_end_image, kf_start_strength, kf_end_strength,
                           kf_prompt, kf_negative_prompt, kf_seed, kf_height, kf_width,
                           kf_num_frames, kf_fps, kf_steps, kf_cfg_scale, kf_enhance_prompt],
                    outputs=kf_output
                )
        
        # 底部说明
        gr.Markdown("""
        ---
        ### 💡 提示词编写建议
        - 从主要动作开始，用一句话描述
        - 添加具体的动作和手势细节
        - 精确描述角色/物体外观
        - 包含背景和环境细节
        - 指定相机角度和运动
        - 描述光照和色彩
        - 保持在200字以内
        
        ### 📊 技术规格
        - **帧数**: 必须是 1+8k 格式 (17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, ..., 257)
        - **分辨率**: 宽高必须是 32 的倍数
        - **最大时长**: 257帧 ≈ 10.7秒 @24fps
        
        ### 📚 更多信息
        - [LTX-2 官方文档](https://github.com/Lightricks/LTX-2)
        - [HuggingFace 模型](https://huggingface.co/Lightricks/LTX-2)
        - [提示词编写指南](https://ltx.video/blog/how-to-prompt-for-ltx-2)
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
