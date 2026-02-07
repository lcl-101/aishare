import os
import sys
import random
import logging
import tempfile
from datetime import datetime

import gradio as gr
import torch
import numpy as np
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import save_video
import imageio.v2 as imageio
from PIL import ImageDraw, ImageFont

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# 全局模型变量
wan_i2v = None

# 检查点路径
CKPT_DIR = "checkpoints/lingbot-world-base-cam"

# 示例数据
EXAMPLES = [
    {
        "name": "示例 00 - 奇幻丛林飞行",
        "image": "examples/00/image.jpg",
        "action_path": "examples/00",
        "prompt": "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."
    },
    {
        "name": "示例 01 - 巨石阵全景",
        "image": "examples/01/image.jpg",
        "action_path": "examples/01",
        "prompt": "A slow panoramic sweep around Stonehenge on a misty, overcast day, capturing the ancient standing stones in serene stillness, with soft ambient wind and distant bird calls enhancing the timeless atmosphere."
    },
    {
        "name": "示例 02 - 城市漫游",
        "image": "examples/02/image.jpg",
        "action_path": "examples/02",
        "prompt": "The video presents a cinematic, first-person wandering experience through a hyper-realistic urban environment rendered in a video game engine. It begins with a static, sun-drenched alley framed by graffiti-laden industrial walls and overhead power lines, immediately establishing a gritty, lived-in atmosphere. As the camera pans right and tilts upward, it reveals a sprawling cityscape dominated by towering skyscrapers and industrial infrastructure, all bathed in warm, late-afternoon light that casts long shadows and produces dramatic lens flares. The perspective then transitions into a smooth forward tracking shot along a cracked sidewalk, passing weathered fences, palm trees, and distant pedestrians, creating a sense of immersion and exploration. Midway, the camera briefly follows a walking figure before refocusing on the broader streetscape, culminating in a stabilized view of a small blue van parked at an intersection surrounded by urban elements like parking garages and traffic lights. The entire sequence is characterized by its photorealistic detail, dynamic lighting, and deliberate pacing, evoking the feel of a quiet, sunlit afternoon in a futuristic metropolis."
    }
]


def load_model():
    """加载模型"""
    global wan_i2v
    if wan_i2v is not None:
        return "模型已加载"
    
    logging.info("正在加载模型...")
    cfg = WAN_CONFIGS["i2v-A14B"]
    
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=CKPT_DIR,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=False,
    )
    logging.info("模型加载完成")
    return "模型加载完成"


def _infer_wasd_from_poses(poses, frame_num):
    trans = poses[:, :3, 3]
    dirs = []
    for i in range(len(trans) - 1):
        world_delta = trans[i + 1] - trans[i]
        R = poses[i, :3, :3]
        local_delta = R.T.dot(world_delta)
        dx, dy, dz = local_delta
        if abs(dx) > abs(dz):
            dirs.append('D' if dx > 0 else 'A')
        else:
            dirs.append('W' if dz < 0 else 'S')
    if len(dirs) > 0:
        dirs.append(dirs[-1])
    if len(dirs) < int(frame_num):
        dirs.extend([dirs[-1] if dirs else ''] * (int(frame_num) - len(dirs)))
    return dirs


def _overlay_wasd_on_video(input_file, action_path, frame_num, fps):
    poses_path = os.path.join(action_path, "poses.npy")
    if not os.path.exists(poses_path):
        return input_file
    poses = np.load(poses_path)
    dirs = _infer_wasd_from_poses(poses, frame_num)

    overlay_file = input_file.replace('.mp4', '_wasd.mp4')
    reader = imageio.get_reader(input_file)
    writer = imageio.get_writer(overlay_file, fps=fps)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, frame in enumerate(reader):
        img_pil = Image.fromarray(frame).convert("RGBA")
        draw = ImageDraw.Draw(img_pil)
        label = dirs[i] if i < len(dirs) else ''
        if label:
            text = f"{label}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            pad = 8
            rect_xy = (10, 10, 10 + text_w + pad, 10 + text_h + pad)
            draw.rectangle(rect_xy, fill=(0, 0, 0, 140))
            draw.text((14, 12), text, fill=(255, 235, 59, 255), font=font)
        writer.append_data(np.array(img_pil.convert("RGB")))
    writer.close()
    reader.close()
    return overlay_file


def generate_video(
    image,
    prompt,
    action_path,
    size,
    frame_num,
    sample_steps,
    sample_shift,
    guide_scale,
    seed,
    progress=gr.Progress()
):
    """生成视频"""
    global wan_i2v
    
    if wan_i2v is None:
        load_model()
    
    if image is None:
        return None, "请上传图片"
    
    if not prompt:
        return None, "请输入提示词"
    
    if not action_path or not os.path.exists(action_path):
        return None, "请选择有效的相机轨迹路径"
    
    # 处理种子
    if seed < 0:
        seed = random.randint(0, sys.maxsize)
    
    logging.info(f"开始生成视频...")
    logging.info(f"提示词: {prompt}")
    logging.info(f"相机轨迹: {action_path}")
    logging.info(f"尺寸: {size}")
    logging.info(f"帧数: {frame_num}")
    logging.info(f"种子: {seed}")
    
    try:
        # 转换图片
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = Image.fromarray(image).convert("RGB")
        
        cfg = WAN_CONFIGS["i2v-A14B"]
        
        # 生成视频
        video = wan_i2v.generate(
            prompt,
            img,
            action_path=action_path,
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=int(frame_num),
            shift=float(sample_shift),
            sample_solver='unipc',
            sampling_steps=int(sample_steps),
            guide_scale=float(guide_scale),
            seed=int(seed),
            offload_model=True
        )
        
        # 保存视频
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(tempfile.gettempdir(), f"lingbot_world_{formatted_time}.mp4")
        
        save_video(
            tensor=video[None],
            save_file=output_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

        # 在视频上叠加 WASD 风格的方向标记（基于 poses.npy 推断移动方向）
        try:
            final_output = _overlay_wasd_on_video(
                input_file=output_file,
                action_path=action_path,
                frame_num=int(frame_num),
                fps=cfg.sample_fps,
            )
        except Exception as e:
            logging.warning(f"叠加 WASD 标记失败: {e}")
            final_output = output_file

        del video
        torch.cuda.empty_cache()

        logging.info(f"视频生成完成: {final_output}")
        return final_output, f"生成成功！种子: {seed}\n输出文件: {final_output}"
    
    except Exception as e:
        logging.error(f"生成失败: {str(e)}")
        return None, f"生成失败: {str(e)}"


def load_example(example_name):
    """加载示例"""
    for example in EXAMPLES:
        if example["name"] == example_name:
            return (
                example["image"],
                example["prompt"],
                example["action_path"]
            )
    return None, "", ""


def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="LingBot-World 视频生成", theme=gr.themes.Soft()) as demo:
        # 顶部频道信息
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">🎬 LingBot-World 视频生成</h1>
            <p style="color: #f0f0f0; margin-top: 10px;">
                📺 欢迎访问我的 YouTube 频道: 
                <a href="https://www.youtube.com/@rongyi-ai" target="_blank" style="color: #ffeb3b; font-weight: bold;">
                    AI 技术分享频道
                </a>
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📥 输入设置")
                
                # 示例选择
                example_dropdown = gr.Dropdown(
                    choices=[e["name"] for e in EXAMPLES],
                    label="选择示例",
                    info="选择预设示例快速体验"
                )
                
                # 图片上传
                image_input = gr.Image(
                    label="输入图片",
                    type="filepath",
                    height=300
                )
                
                # 提示词
                prompt_input = gr.Textbox(
                    label="提示词 (Prompt)",
                    placeholder="请输入描述视频内容的提示词...",
                    lines=5
                )
                
                # 相机轨迹路径
                action_path_input = gr.Textbox(
                    label="相机轨迹路径",
                    placeholder="例如: examples/00",
                    info="包含 poses.npy 和 intrinsics.npy 的目录路径"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 生成参数")
                
                # 尺寸选择
                size_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_SIZES["i2v-A14B"]),
                    value="480*832",
                    label="输出尺寸",
                    info="视频分辨率 (宽*高)"
                )
                
                # 帧数
                frame_num_slider = gr.Slider(
                    minimum=17,
                    maximum=161,
                    step=4,
                    value=81,
                    label="帧数",
                    info="生成的视频帧数，必须是 4n+1 格式"
                )
                
                # 采样步数
                sample_steps_slider = gr.Slider(
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=30,
                    label="采样步数",
                    info="更多步数=更高质量，但更慢"
                )
                
                # Shift 参数
                shift_slider = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=3.0,
                    label="Shift 参数",
                    info="采样偏移因子"
                )
                
                # 引导系数
                guide_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.5,
                    value=5.0,
                    label="引导系数 (CFG Scale)",
                    info="更高=更遵循提示词"
                )
                
                # 随机种子
                seed_input = gr.Number(
                    value=42,
                    label="随机种子",
                    info="设为 -1 使用随机种子"
                )
                
                # 生成按钮
                generate_btn = gr.Button(
                    "🚀 生成视频",
                    variant="primary",
                    size="lg"
                )
        
        gr.Markdown("## 📤 输出结果")
        
        with gr.Row():
            with gr.Column():
                video_output = gr.Video(
                    label="生成的视频",
                    height=400
                )
                status_output = gr.Textbox(
                    label="状态信息",
                    interactive=False
                )
        
        # 事件绑定
        example_dropdown.change(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[image_input, prompt_input, action_path_input]
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                image_input,
                prompt_input,
                action_path_input,
                size_dropdown,
                frame_num_slider,
                sample_steps_slider,
                shift_slider,
                guide_scale_slider,
                seed_input
            ],
            outputs=[video_output, status_output]
        )
        
        # 使用说明
        gr.Markdown("""
        ---
        ## 📖 使用说明
        
        1. **选择示例**: 从下拉菜单选择预设示例，或手动上传图片和设置参数
        2. **上传图片**: 上传作为视频起始帧的图片
        3. **输入提示词**: 描述您想要生成的视频内容（建议使用英文）
        4. **设置相机轨迹**: 指定包含相机运动数据的目录
        5. **调整参数**: 根据需要调整分辨率、帧数等参数
        6. **点击生成**: 等待视频生成完成
        
        ⚠️ **注意**: 首次生成时需要加载模型，可能需要较长时间。
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
