import gc
import logging
import os
import random
import subprocess
import time

import gradio as gr
import imageio
import numpy as np
import requests
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler()],
)


class _ListWithDevice(list):
    """list 子类，添加 .device 属性以兼容 x.device 访问"""
    @property
    def device(self):
        return self[0].device if len(self) > 0 else torch.device("cpu")


def patch_transformer_forward(transformer):
    """
    修复 WanModel.forward 中 x 为 list 时 x.device 报错的问题。
    原代码第 738 行: if not block_offload and self.freqs.device != x.device and isinstance(x, torch.Tensor)
    短路求值顺序有误，x.device 在 isinstance 检查之前就被执行了。
    通过将 list 包装为带 .device 属性的子类来避免崩溃。
    """
    cls = type(transformer)
    if getattr(cls, '_forward_patched', False):
        return

    _original_forward = cls.forward

    def _patched_forward(self, x, *args, **kwargs):
        if isinstance(x, list) and not isinstance(x, _ListWithDevice):
            x = _ListWithDevice(x)
        return _original_forward(self, x, *args, **kwargs)

    cls.forward = _patched_forward
    cls._forward_patched = True

# ============ 模型路径 ============
MODEL_PATH_R2V = "checkpoints/SkyReels-V3-R2V-14B"
MODEL_PATH_V2V = "checkpoints/SkyReels-V3-V2V-14B"
MODEL_PATH_A2V = "checkpoints/SkyReels-V3-A2V-19B"

# ============ 全局 pipeline 缓存（按需加载） ============
_pipelines = {}

# ============ 示例资源下载 ============
EXAMPLE_DIR = "assets/examples"
os.makedirs(EXAMPLE_DIR, exist_ok=True)

EXAMPLE_URLS = {
    "ref_1.png": "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_1.png",
    "ref_2.png": "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_2.png",
    "test_video.mp4": "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4",
    "avatar_woman.JPEG": "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/woman.JPEG",
    "avatar_woman_speech.mp3": "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single_actor/woman_speech.mp3",
}


def download_example_file(filename, url):
    filepath = os.path.join(EXAMPLE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"正在下载示例文件: {filename}")
        r = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"下载完成: {filename}")
    return filepath


print("正在检查并下载示例资源...")
for filename, url in EXAMPLE_URLS.items():
    download_example_file(filename, url)
print("示例资源准备完成！")

# ============ 示例提示词 ============
PROMPT_REF_TO_VIDEO = "In a dimly lit, cluttered occult club room adorned with shelves full of books, skulls, and mysterious dolls, two young Asian girls are talking. One girl has vibrant teal pigtails with bangs, wearing a white collared polo shirt, while the other has a sleek black bob with bangs, also in a white polo shirt, conversing under the hum of fluorescent lights, a high-quality and detailed cinematic shot."

PROMPT_SINGLE_SHOT = "A man is making his way forward slowly, leaning on a white cane to prop himself up."

PROMPT_SHOT_SWITCHING = "[ZOOM_IN_CUT] The scene cuts from a medium shot of a visually impaired man walking on a path in a park. The shot then cut in to a close-up of the man's face and upper torso. The visually impaired Black man is shown from the chest up, wearing dark sunglasses, a grey turtleneck scarf, and a light olive green jacket. His head is held straight, looking forward towards the camera, continuing his walk. The lighting is natural and bright. The background is a soft blur of green trees and foliage from the park."

PROMPT_TALKING_AVATAR = "A woman is giving a speech. She is confident, poised, and joyful. Use a static shot."


# ============ Pipeline 加载/卸载 ============
def unload_all_pipelines():
    """卸载所有已加载的 pipeline，释放显存"""
    global _pipelines
    for name in list(_pipelines.keys()):
        del _pipelines[name]
    _pipelines.clear()
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("已卸载所有 pipeline，释放显存")


def get_r2v_pipeline():
    global _pipelines
    if "r2v" not in _pipelines:
        unload_all_pipelines()
        from skyreels_v3.pipelines import ReferenceToVideoPipeline
        logging.info("正在加载 Reference-to-Video pipeline...")
        _pipelines["r2v"] = ReferenceToVideoPipeline(
            model_path=MODEL_PATH_R2V,
            offload=True,
        )
        logging.info("Reference-to-Video pipeline 加载完成")
    return _pipelines["r2v"]


def get_single_shot_pipeline():
    global _pipelines
    if "single_shot" not in _pipelines:
        unload_all_pipelines()
        from skyreels_v3.pipelines import SingleShotExtensionPipeline
        logging.info("正在加载 Single-shot Extension pipeline...")
        _pipelines["single_shot"] = SingleShotExtensionPipeline(
            model_path=MODEL_PATH_V2V,
            offload=True,
        )
        patch_transformer_forward(_pipelines["single_shot"].transformer)
        logging.info("Single-shot Extension pipeline 加载完成")
    return _pipelines["single_shot"]


def get_shot_switching_pipeline():
    global _pipelines
    if "shot_switching" not in _pipelines:
        unload_all_pipelines()
        from skyreels_v3.pipelines import ShotSwitchingExtensionPipeline
        logging.info("正在加载 Shot Switching Extension pipeline...")
        _pipelines["shot_switching"] = ShotSwitchingExtensionPipeline(
            model_path=MODEL_PATH_V2V,
            offload=True,
        )
        patch_transformer_forward(_pipelines["shot_switching"].transformer)
        logging.info("Shot Switching Extension pipeline 加载完成")
    return _pipelines["shot_switching"]


def get_talking_avatar_pipeline():
    global _pipelines
    if "talking_avatar" not in _pipelines:
        unload_all_pipelines()
        from skyreels_v3.configs import WAN_CONFIGS
        from skyreels_v3.pipelines import TalkingAvatarPipeline
        config = WAN_CONFIGS["talking-avatar-19B"]
        logging.info("正在加载 Talking Avatar pipeline...")
        _pipelines["talking_avatar"] = TalkingAvatarPipeline(
            config=config,
            model_path=MODEL_PATH_A2V,
            device_id=0,
            rank=0,
            offload=True,
        )
        logging.info("Talking Avatar pipeline 加载完成")
    return _pipelines["talking_avatar"]


# ============ 保存视频 ============
def save_video(video_frames, task_type, seed, fps=24, input_data=None):
    """保存视频帧为 mp4 文件，返回文件路径"""
    save_dir = os.path.join("result", task_type)
    os.makedirs(save_dir, exist_ok=True)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_out_file = f"{seed}_{current_time}.mp4"
    output_path = os.path.join(save_dir, video_out_file)

    imageio.mimwrite(
        output_path,
        video_frames,
        fps=fps,
        quality=8,
        output_params=["-loglevel", "error"],
    )

    # 对于 talking_avatar，合并音频
    if task_type == "talking_avatar" and input_data is not None:
        video_with_audio_path = os.path.join(save_dir, video_out_file.replace(".mp4", "_with_audio.mp4"))
        audio_path = input_data["video_audio"]
        video_in = os.path.abspath(output_path)
        audio_in = os.path.abspath(audio_path)
        video_out_with_audio = os.path.abspath(video_with_audio_path)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", audio_in,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-shortest",
            video_out_with_audio,
        ]
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            logging.info(f"带音频视频生成成功: {video_with_audio_path}")
            os.remove(video_in)
            return video_with_audio_path
        except subprocess.CalledProcessError as e:
            logging.error(f"ffmpeg 合并音频失败: {e.stdout}")
            return output_path

    return output_path


# ============ 参考图生成视频 ============
def reference_to_video(ref_img1, ref_img2, ref_img3, ref_img4, prompt, duration, seed, progress=gr.Progress()):
    ref_img_paths = [img for img in [ref_img1, ref_img2, ref_img3, ref_img4] if img is not None and img != ""]
    if len(ref_img_paths) == 0:
        return None, "❌ 错误：请至少上传一张参考图片"

    try:
        progress(0.1, desc="正在加载模型...")
        pipe = get_r2v_pipeline()

        # 加载参考图片
        ref_imgs = [load_image(p) for p in ref_img_paths]

        progress(0.3, desc="正在生成视频...")
        video_frames = pipe.generate_video(
            ref_imgs=ref_imgs,
            prompt=prompt,
            duration=int(duration),
            seed=int(seed),
        )

        progress(0.9, desc="正在保存视频...")
        output_path = save_video(video_frames, "reference_to_video", int(seed), fps=24)
        return output_path, f"✅ 生成成功！视频保存至: {output_path}"
    except Exception as e:
        logging.exception("参考图生成视频失败")
        return None, f"❌ 生成失败: {str(e)}"


# ============ 单段视频扩展 ============
def single_shot_extension(input_video, prompt, duration, seed, progress=gr.Progress()):
    if input_video is None or input_video == "":
        return None, "❌ 错误：请上传输入视频"

    try:
        progress(0.1, desc="正在加载模型...")
        pipe = get_single_shot_pipeline()

        progress(0.3, desc="正在扩展视频...")
        video_frames = pipe.extend_video(
            raw_video=input_video,
            prompt=prompt,
            duration=int(duration),
            seed=int(seed),
        )

        progress(0.9, desc="正在保存视频...")
        output_path = save_video(video_frames, "single_shot_extension", int(seed), fps=24)
        return output_path, f"✅ 生成成功！视频保存至: {output_path}"
    except Exception as e:
        logging.exception("单段视频扩展失败")
        return None, f"❌ 生成失败: {str(e)}"


# ============ 镜头切换扩展 ============
def shot_switching_extension(input_video, prompt, seed, progress=gr.Progress()):
    if input_video is None or input_video == "":
        return None, "❌ 错误：请上传输入视频"

    try:
        progress(0.1, desc="正在加载模型...")
        pipe = get_shot_switching_pipeline()

        progress(0.3, desc="正在生成视频...")
        video_frames = pipe.extend_video(
            raw_video=input_video,
            prompt=prompt,
            duration=5,
            seed=int(seed),
        )

        progress(0.9, desc="正在保存视频...")
        output_path = save_video(video_frames, "shot_switching_extension", int(seed), fps=24)
        return output_path, f"✅ 生成成功！视频保存至: {output_path}"
    except Exception as e:
        logging.exception("镜头切换扩展失败")
        return None, f"❌ 生成失败: {str(e)}"


# ============ 说话头像生成 ============
def talking_avatar(input_image, input_audio, prompt, seed, progress=gr.Progress()):
    if input_image is None or input_image == "":
        return None, "❌ 错误：请上传肖像图片"
    if input_audio is None or input_audio == "":
        return None, "❌ 错误：请上传驱动音频"

    try:
        progress(0.1, desc="正在加载模型...")
        pipe = get_talking_avatar_pipeline()

        # 准备输入数据
        input_data = {
            "prompt": prompt,
            "cond_image": input_image,
            "cond_audio": {"person1": input_audio},
        }

        progress(0.2, desc="正在预处理音频...")
        from skyreels_v3.utils.avatar_preprocess import preprocess_audio
        input_data, _ = preprocess_audio(MODEL_PATH_A2V, input_data, "processed_audio")

        progress(0.3, desc="正在生成视频...")
        kwargs = {
            "input_data": input_data,
            "size_buckget": "720P",
            "motion_frame": 5,
            "frame_num": 81,
            "drop_frame": 12,
            "shift": 11,
            "text_guide_scale": 1.0,
            "audio_guide_scale": 1.0,
            "seed": int(seed),
            "sampling_steps": 4,
            "max_frames_num": 5000,
        }
        video_frames = pipe.generate(**kwargs)

        progress(0.9, desc="正在保存视频...")
        output_path = save_video(video_frames, "talking_avatar", int(seed), fps=25, input_data=input_data)
        return output_path, f"✅ 生成成功！视频保存至: {output_path}"
    except Exception as e:
        logging.exception("说话头像生成失败")
        return None, f"❌ 生成失败: {str(e)}"


# ============ 使用示例 ============
def use_example_ref_imgs():
    return (
        os.path.join(EXAMPLE_DIR, "ref_1.png"),
        os.path.join(EXAMPLE_DIR, "ref_2.png"),
        None,
        None,
    )


def use_example_video():
    return os.path.join(EXAMPLE_DIR, "test_video.mp4")


def use_example_avatar():
    return (
        os.path.join(EXAMPLE_DIR, "avatar_woman.JPEG"),
        os.path.join(EXAMPLE_DIR, "avatar_woman_speech.mp3"),
    )


# ============ Gradio 界面 ============
with gr.Blocks(title="SkyReels-V3 视频生成") as demo:

    # 顶部 YouTube 频道信息
    gr.HTML("""
    <div style='text-align:center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>🎬 SkyReels-V3 视频生成</h1>
        <p style='color: white; margin: 10px 0 0 0; font-size: 16px;'>
            📺 <b>AI 技术分享频道</b> |
            <a href='https://www.youtube.com/@rongyi-ai' target='_blank' style='color: #ffeb3b; text-decoration: none;'>
                https://www.youtube.com/@rongyi-ai
            </a>
        </p>
    </div>
    """)

    with gr.Tabs():
        # ============ Tab 1: 参考图生成视频 ============
        with gr.TabItem("📷 参考图生成视频"):
            gr.Markdown("""
            ### 参考图生成视频 (Reference-to-Video)
            从1-4张参考图片和文字提示生成连贯的视频序列，擅长保持角色、物体和背景的身份一致性。
            - **推荐输出**: 5秒视频，720p，24fps
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    ref_img1 = gr.Image(label="参考图片 1", type="filepath")
                    ref_img2 = gr.Image(label="参考图片 2", type="filepath")
                with gr.Column(scale=1):
                    ref_img3 = gr.Image(label="参考图片 3 (可选)", type="filepath")
                    ref_img4 = gr.Image(label="参考图片 4 (可选)", type="filepath")

            example_btn1 = gr.Button("📥 使用示例图片", variant="secondary")

            prompt_ref = gr.Textbox(label="提示词", value=PROMPT_REF_TO_VIDEO, lines=4)

            with gr.Row():
                duration_ref = gr.Slider(label="视频时长 (秒)", minimum=1, maximum=5, value=5, step=1)
                seed_ref = gr.Number(label="随机种子", value=42)

            btn_ref = gr.Button("🎬 生成视频", variant="primary", size="lg")
            status_ref = gr.Textbox(label="状态", interactive=False)
            output_ref = gr.Video(label="生成结果")

            example_btn1.click(use_example_ref_imgs, outputs=[ref_img1, ref_img2, ref_img3, ref_img4])
            btn_ref.click(
                reference_to_video,
                inputs=[ref_img1, ref_img2, ref_img3, ref_img4, prompt_ref, duration_ref, seed_ref],
                outputs=[output_ref, status_ref],
            )

        # ============ Tab 2: 单段视频扩展 ============
        with gr.TabItem("🎞️ 单段视频扩展"):
            gr.Markdown("""
            ### 单段视频扩展 (Single-shot Extension)
            扩展现有视频，同时保持运动连续性、场景一致性和主体身份。
            - **扩展时长**: 5-30秒
            """)

            input_video_single = gr.Video(label="输入视频")
            example_btn2 = gr.Button("📥 使用示例视频", variant="secondary")

            prompt_single = gr.Textbox(label="提示词", value=PROMPT_SINGLE_SHOT, lines=3)

            with gr.Row():
                duration_single = gr.Slider(label="扩展时长 (秒)", minimum=5, maximum=30, value=5, step=1)
                seed_single = gr.Number(label="随机种子", value=42)

            btn_single = gr.Button("🎬 扩展视频", variant="primary", size="lg")
            status_single = gr.Textbox(label="状态", interactive=False)
            output_single = gr.Video(label="生成结果")

            example_btn2.click(use_example_video, outputs=input_video_single)
            btn_single.click(
                single_shot_extension,
                inputs=[input_video_single, prompt_single, duration_single, seed_single],
                outputs=[output_single, status_single],
            )

        # ============ Tab 3: 镜头切换扩展 ============
        with gr.TabItem("🎥 镜头切换扩展"):
            gr.Markdown("""
            ### 镜头切换扩展 (Shot Switching Extension)
            支持电影级镜头转换，如推进 (Cut-In)、拉远 (Cut-Out)、正反打 (Shot/Reverse Shot) 等。
            - **最大时长**: 5秒
            - **提示词前缀**: `[ZOOM_IN_CUT]`, `[ZOOM_OUT_CUT]`, `[SHOT_REVERSE_SHOT]`, `[MULTI_ANGLE]`, `[CUT_AWAY]`
            """)

            input_video_switch = gr.Video(label="输入视频")
            example_btn3 = gr.Button("📥 使用示例视频", variant="secondary")

            prompt_switch = gr.Textbox(label="提示词", value=PROMPT_SHOT_SWITCHING, lines=5)

            seed_switch = gr.Number(label="随机种子", value=42)

            btn_switch = gr.Button("🎬 生成视频", variant="primary", size="lg")
            status_switch = gr.Textbox(label="状态", interactive=False)
            output_switch = gr.Video(label="生成结果")

            example_btn3.click(use_example_video, outputs=input_video_switch)
            btn_switch.click(
                shot_switching_extension,
                inputs=[input_video_switch, prompt_switch, seed_switch],
                outputs=[output_switch, status_switch],
            )

        # ============ Tab 4: 说话头像生成 ============
        with gr.TabItem("🗣️ 说话头像生成"):
            gr.Markdown("""
            ### 说话头像生成 (Talking Avatar)
            从单张肖像和音频片段生成逼真的说话头像视频。
            - **支持图片格式**: jpg/jpeg, png, gif, bmp
            - **支持音频格式**: mp3, wav
            - **最大音频时长**: 200秒
            """)

            with gr.Row():
                input_image_avatar = gr.Image(label="肖像图片", type="filepath")
                input_audio_avatar = gr.Audio(label="驱动音频", type="filepath")

            example_btn4 = gr.Button("📥 使用示例图片和音频", variant="secondary")

            prompt_avatar = gr.Textbox(label="提示词", value=PROMPT_TALKING_AVATAR, lines=2)

            seed_avatar = gr.Number(label="随机种子", value=42)

            btn_avatar = gr.Button("🎬 生成视频", variant="primary", size="lg")
            status_avatar = gr.Textbox(label="状态", interactive=False)
            output_avatar = gr.Video(label="生成结果")

            example_btn4.click(use_example_avatar, outputs=[input_image_avatar, input_audio_avatar])
            btn_avatar.click(
                talking_avatar,
                inputs=[input_image_avatar, input_audio_avatar, prompt_avatar, seed_avatar],
                outputs=[output_avatar, status_avatar],
            )

    gr.Markdown("""
    ---
    <div style='text-align: center; color: #666;'>
        基于 <b>SkyReels-V3</b> | 模型路径: checkpoints/
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
