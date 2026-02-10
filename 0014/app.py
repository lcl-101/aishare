import glob
import os
import tempfile
import traceback
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from decord import VideoReader
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from einops import rearrange
from omegaconf import OmegaConf

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from telestylevideo_pipeline import WanPipeline
from telestylevideo_transformer import WanTransformer3DModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
TELESTYLE_DIR = os.path.join(CHECKPOINTS_DIR, "TeleStyle")
TELESTYLE_WEIGHTS_DIR = os.path.join(TELESTYLE_DIR, "weights")
WAN_T2V_DIR = os.path.join(CHECKPOINTS_DIR, "Wan2.1-T2V-1.3B-Diffusers")
QWEN_IMAGE_EDIT_DIR = os.path.join(CHECKPOINTS_DIR, "Qwen-Image-Edit-2509")

DEFAULT_PROMPT = (
    "Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1."
)

_image_engine = None
_video_engine = None


def _ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, str) and os.path.exists(img):
        return Image.open(img)
    raise ValueError("无法读取输入图像")


class LocalImageStyleInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        device_str = "cuda" if self.device.type == "cuda" else "cpu"

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=dtype,
            device=device_str,
            model_configs=[
                ModelConfig(
                    path=sorted(glob.glob(os.path.join(QWEN_IMAGE_EDIT_DIR, "transformer", "diffusion_pytorch_model*.safetensors"))),
                ),
                ModelConfig(
                    path=sorted(glob.glob(os.path.join(QWEN_IMAGE_EDIT_DIR, "text_encoder", "model*.safetensors"))),
                ),
                ModelConfig(
                    path=os.path.join(QWEN_IMAGE_EDIT_DIR, "vae", "diffusion_pytorch_model.safetensors"),
                ),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(
                path=os.path.join(QWEN_IMAGE_EDIT_DIR, "processor"),
            ),
        )

        telestyle_path = os.path.join(TELESTYLE_WEIGHTS_DIR, "diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors")
        speedup_path = os.path.join(TELESTYLE_WEIGHTS_DIR, "diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")

        self.pipe.load_lora(self.pipe.dit, telestyle_path)
        self.pipe.load_lora(self.pipe.dit, speedup_path)

    def inference(
        self,
        prompt: str,
        content_ref: Image.Image,
        style_ref: Image.Image,
        seed: int = 123,
        num_inference_steps: int = 4,
        minedge: int = 1024,
    ) -> Image.Image:
        content_ref = _ensure_pil(content_ref).convert("RGB")
        style_ref = _ensure_pil(style_ref).convert("RGB")

        w, h = content_ref.size
        minedge = minedge - minedge % 16

        if w > h:
            r = w / h
            h = minedge
            w = int(h * r) - int(h * r) % 16
        else:
            r = h / w
            w = minedge
            h = int(w * r) - int(w * r) % 16

        images = [
            content_ref.resize((w, h)),
            style_ref.resize((minedge, minedge)),
        ]

        result = self.pipe(
            prompt,
            edit_image=images,
            seed=seed,
            num_inference_steps=num_inference_steps,
            height=h,
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=1.0,
        )

        return result


class LocalVideoStyleInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
        self.update_settings(42, 129, 720, 1248, 25)

    def _load_models(self):
        self.vae_path = os.path.join(WAN_T2V_DIR, "vae")
        self.transformer_config_path = os.path.join(WAN_T2V_DIR, "transformer", "config.json")
        self.scheduler_path = os.path.join(WAN_T2V_DIR, "scheduler")
        self.ckpt_path = os.path.join(TELESTYLE_WEIGHTS_DIR, "dit.ckpt")
        self.prompt_embeds_path = os.path.join(TELESTYLE_WEIGHTS_DIR, "prompt_embeds.pth")

        state_dict = torch.load(self.ckpt_path, map_location="cpu")["transformer_state_dict"]
        transformer_state_dict = {}
        for key in state_dict:
            transformer_state_dict[key.split("module.")[1]] = state_dict[key]

        config = OmegaConf.to_container(OmegaConf.load(self.transformer_config_path))

        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.vae = AutoencoderKLWan.from_pretrained(self.vae_path, torch_dtype=self.dtype).to(self.device)
        self.transformer = WanTransformer3DModel(**config)
        self.transformer.load_state_dict(transformer_state_dict)
        self.transformer = self.transformer.to(self.device).to(self.dtype)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.scheduler_path)

        self.pipe = WanPipeline(
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
        )
        self.pipe.to(self.device)
        self.prompt_embeds_ = torch.load(self.prompt_embeds_path, map_location="cpu")

    def update_settings(
        self,
        random_seed: int,
        video_length: int,
        height: int,
        width: int,
        num_inference_steps: int,
    ):
        self.random_seed = int(random_seed)
        self.video_length = int(video_length)
        self.H = int(height)
        self.W = int(width)
        self.num_inference_steps = int(num_inference_steps)

    @staticmethod
    def load_video(video_path: str, video_length: int) -> torch.Tensor:
        if any(ext in video_path.lower() for ext in [".png", ".jpeg", ".jpg"]):
            image = Image.open(video_path).convert("RGB")
            image = np.array(image)
            image = image[None, None]
            image = torch.from_numpy(image) / 127.5 - 1.0
            return image

        vr = VideoReader(video_path)
        frames = list(range(min(len(vr), video_length)))
        images = vr.get_batch(frames).asnumpy()
        images = torch.from_numpy(images) / 127.5 - 1.0
        images = images[None]
        return images

    def inference(self, source_videos: torch.Tensor, first_images: torch.Tensor) -> torch.Tensor:
        source_videos = source_videos.to(self.device).to(self.dtype)
        first_images = first_images.to(self.device).to(self.dtype)
        prompt_embeds_ = self.prompt_embeds_.to(self.device).to(self.dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean)
        latents_mean = latents_mean.view(1, 16, 1, 1, 1).to(self.device, self.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std)
        latents_std = latents_std.view(1, 16, 1, 1, 1).to(self.device, self.dtype)

        bsz = 1
        _, _, h, w, _ = source_videos.shape

        if h < w:
            output_h, output_w = self.H, self.W
        else:
            output_h, output_w = self.W, self.H

        with torch.no_grad():
            source_videos = rearrange(source_videos, "b f h w c -> (b f) c h w")
            source_videos = F.interpolate(source_videos, (output_h, output_w), mode="bilinear")
            source_videos = rearrange(source_videos, "(b f) c h w -> b c f h w", b=bsz)

            first_images = rearrange(first_images, "b f h w c -> (b f) c h w")
            first_images = F.interpolate(first_images, (output_h, output_w), mode="bilinear")
            first_images = rearrange(first_images, "(b f) c h w -> b c f h w", b=bsz)

            source_latents = self.vae.encode(source_videos).latent_dist.mode()
            source_latents = (source_latents - latents_mean) * latents_std

            first_latents = self.vae.encode(first_images).latent_dist.mode()
            first_latents = (first_latents - latents_mean) * latents_std

            neg_first_latents = self.vae.encode(torch.zeros_like(first_images)).latent_dist.mode()
            neg_first_latents = (neg_first_latents - latents_mean) * latents_std

        video = self.pipe(
            source_latents=source_latents,
            first_latents=first_latents,
            neg_first_latents=neg_first_latents,
            num_frames=self.video_length,
            guidance_scale=1.0,
            height=output_h,
            width=output_w,
            prompt_embeds_=prompt_embeds_,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(self.random_seed),
        ).frames[0]

        return video


def get_image_engine() -> LocalImageStyleInference:
    global _image_engine
    if _image_engine is None:
        _image_engine = LocalImageStyleInference()
    return _image_engine


def get_video_engine() -> LocalVideoStyleInference:
    global _video_engine
    if _video_engine is None:
        _video_engine = LocalVideoStyleInference()
    return _video_engine


def auto_detect_video(video_path):
    """根据上传的视频自动识别帧数、高度、宽度、帧率"""
    if not video_path:
        return gr.update(), gr.update(), gr.update(), gr.update()
    try:
        vr = VideoReader(video_path)
        num_frames = len(vr)
        fps = round(vr.get_avg_fps())
        h, w = vr[0].shape[:2]
        # 对齐到 16 的倍数
        h = h - h % 16
        w = w - w % 16
        print(f"[视频检测] 帧数: {num_frames}, 帧率: {fps}, 高度: {h}, 宽度: {w}")
        return gr.update(value=num_frames), gr.update(value=h), gr.update(value=w), gr.update(value=fps)
    except Exception as e:
        print(f"[视频检测] 无法读取视频信息: {e}")
        return gr.update(), gr.update(), gr.update(), gr.update()


def run_image(prompt, content_image, style_image, seed, steps, minedge):
    if content_image is None or style_image is None:
        return None, "请提供内容图像和风格图像。"

    try:
        print("[图片风格化] 开始加载模型...")
        engine = get_image_engine()
        print("[图片风格化] 模型加载完成，开始推理...")

        with torch.no_grad():
            result = engine.inference(
                prompt=prompt or DEFAULT_PROMPT,
                content_ref=content_image,
                style_ref=style_image,
                seed=int(seed),
                num_inference_steps=int(steps),
                minedge=int(minedge),
            )

        print(f"[图片风格化] 推理完成，结果类型: {type(result)}, 尺寸: {result.size if hasattr(result, 'size') else 'N/A'}")
        return result, "生成完成。"
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"[图片风格化] 出错:\n{err_msg}")
        return None, f"生成失败: {str(e)}"


def run_video(video_path, style_image, seed, video_length, height, width, steps, fps):
    if not video_path:
        return None, "请提供源视频。"
    if style_image is None:
        return None, "请提供风格参考图像。"

    try:
        print("[视频风格化] 开始加载模型...")
        engine = get_video_engine()
        engine.update_settings(seed, video_length, height, width, steps)
        print("[视频风格化] 模型加载完成，开始推理...")

        # 使用页面传入的帧率，若未设置则从源视频读取
        output_fps = int(fps) if fps and int(fps) > 0 else round(VideoReader(video_path).get_avg_fps())
        print(f"[视频风格化] 输出帧率: {output_fps}")

        source_video = engine.load_video(video_path, engine.video_length)

        style_pil = _ensure_pil(style_image).convert("RGB")
        style_np = np.array(style_pil)
        style_tensor = torch.from_numpy(style_np) / 127.5 - 1.0
        style_tensor = style_tensor[None, None, :, :, :]

        with torch.no_grad():
            generated_video = engine.inference(source_video, style_tensor)

        output_dir = tempfile.mkdtemp(prefix="telestyle_video_")
        output_path = os.path.join(output_dir, "generated_video.mp4")
        export_to_video(generated_video, output_path, fps=output_fps)
        print(f"[视频风格化] 生成完成（fps={output_fps}），保存至: {output_path}")

        return output_path, "生成完成。"
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"[视频风格化] 出错:\n{err_msg}")
        return None, f"生成失败: {str(e)}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="TeleStyle Web") as demo:
        gr.Markdown(
            """
# TeleStyle Web

🎥 YouTube 频道：**AI 技术分享频道**  
链接：https://www.youtube.com/@rongyi-ai
"""
        )

        with gr.Tabs():
            with gr.Tab("图片风格化"):
                gr.Markdown("上传内容图像和风格图像，输入提示词后生成风格化图片。")

                with gr.Row():
                    content_image = gr.Image(label="内容图像", type="pil")
                    style_image = gr.Image(label="风格图像", type="pil")

                prompt = gr.Textbox(label="提示词", value=DEFAULT_PROMPT, lines=2)

                with gr.Row():
                    seed = gr.Number(label="随机种子", value=123, precision=0)
                    steps = gr.Slider(label="推理步数", minimum=1, maximum=8, value=4, step=1)
                    minedge = gr.Slider(label="最短边", minimum=512, maximum=1536, value=1024, step=16)

                run_btn = gr.Button("开始生成", variant="primary")
                output_image = gr.Image(label="生成结果")
                status = gr.Textbox(label="状态", interactive=False)

                gr.Examples(
                    examples=[
                        [
                            "assets/example/1-0.png",
                            "assets/example/1-1.png",
                            DEFAULT_PROMPT,
                            123,
                            4,
                            1024,
                        ],
                        [
                            "assets/example/2-0.png",
                            "assets/example/2-1.png",
                            DEFAULT_PROMPT,
                            123,
                            4,
                            1024,
                        ],
                    ],
                    inputs=[content_image, style_image, prompt, seed, steps, minedge],
                )

                run_btn.click(
                    run_image,
                    inputs=[prompt, content_image, style_image, seed, steps, minedge],
                    outputs=[output_image, status],
                )

            with gr.Tab("视频风格化"):
                gr.Markdown("上传源视频和风格参考图像，生成风格化视频。")

                with gr.Row():
                    video_input = gr.Video(label="源视频", sources=["upload"])
                    style_video_image = gr.Image(label="风格参考图像", type="pil")

                with gr.Row():
                    v_seed = gr.Number(label="随机种子", value=42, precision=0)
                    v_length = gr.Slider(label="视频长度（帧数）", minimum=8, maximum=9999, value=129, step=1)
                    v_height = gr.Slider(label="输出高度", minimum=256, maximum=1024, value=720, step=16)
                    v_width = gr.Slider(label="输出宽度", minimum=256, maximum=1536, value=1248, step=16)
                    v_steps = gr.Slider(label="推理步数", minimum=1, maximum=50, value=25, step=1)
                    v_fps = gr.Number(label="输出帧率", value=24, precision=0)

                video_input.change(
                    auto_detect_video,
                    inputs=[video_input],
                    outputs=[v_length, v_height, v_width, v_fps],
                )

                v_run_btn = gr.Button("开始生成", variant="primary")
                output_video = gr.Video(label="生成结果", format="mp4")
                v_status = gr.Textbox(label="状态", interactive=False)

                gr.Examples(
                    examples=[
                        ["assets/example/1.mp4", "assets/example/1-0.png", 42, 129, 720, 1248, 25, 24],
                        ["assets/example/2.mp4", "assets/example/2-0.png", 42, 129, 720, 1248, 25, 24],
                    ],
                    inputs=[video_input, style_video_image, v_seed, v_length, v_height, v_width, v_steps, v_fps],
                )

                v_run_btn.click(
                    run_video,
                    inputs=[video_input, style_video_image, v_seed, v_length, v_height, v_width, v_steps, v_fps],
                    outputs=[output_video, v_status],
                )

        gr.Markdown("模型权重路径已默认指向本地 checkpoints 目录。如需自定义路径，请在 app.py 中调整。")

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
