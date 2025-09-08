import os
import math
import traceback
from typing import List, Optional, Tuple
import numpy as np  # for stacking video frames

import gradio as gr

# Lazy imports for heavy libs to avoid import errors during simple syntax checks
processor = None
model = None
DEVICE = "cuda"
# Prefer a local checkpoint if present
_LOCAL_DEFAULT = "checkpoints/Keye-VL-1_5-8B"
if os.path.isdir(_LOCAL_DEFAULT):
    MODEL_PATH_DEFAULT = _LOCAL_DEFAULT
else:
    MODEL_PATH_DEFAULT = os.environ.get("KEYE_MODEL_PATH", "Kwai-Keye/Keye-VL-1_5-8B")
_CURRENT_MODEL_PATH = None


def _select_device() -> str:
    try:
        import torch  # noqa: F401
        import torch.cuda
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        # Torch might not be installed yet; default to cpu so UI still loads
        return "cpu"


def _normalize_dtype_choice(choice: Optional[str]) -> str:
    if choice is None:
        return "auto"
    c = str(choice).strip().lower()
    if c in ("auto", "自动"):
        return "auto"
    if c in ("bf16", "bfloat16"):
        return "bfloat16"
    if c in ("fp16", "float16", "half"):
        return "float16"
    if c in ("fp32", "float32", "full"):
        return "float32"
    return "auto"


def _load_model(
    model_path: str,
    use_flash_attn: bool = True,
    min_visual_tokens: Optional[int] = None,
    max_visual_tokens: Optional[int] = None,
    dtype_choice: Optional[str] = None,
):
    global processor, model, DEVICE, _CURRENT_MODEL_PATH

    import torch
    from transformers import AutoModel, AutoProcessor

    DEVICE = _select_device()

    # Decide torch dtype based on user choice and backend
    dtype_norm = _normalize_dtype_choice(dtype_choice)
    import torch
    if dtype_norm == "auto":
        # Prefer bf16 on CUDA when using flash-attn; else fall back sensibly
        if use_flash_attn and DEVICE == "cuda":
            torch_dtype = torch.bfloat16
        elif DEVICE == "cuda":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_norm, torch.float32)

    kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if use_flash_attn and DEVICE == "cuda" and torch_dtype in (torch.float16, torch.bfloat16):
        # Best-effort: if flash-attn isn't available, fall back gracefully
        kwargs["attn_implementation"] = "flash_attention_2"

    # Load model if not loaded or path changed
    if model is None or _CURRENT_MODEL_PATH != model_path:
        try:
            model = AutoModel.from_pretrained(model_path, **kwargs)
        except Exception:
            # Retry without flash-attn if it failed
            if "attn_implementation" in kwargs:
                kwargs.pop("attn_implementation", None)
                model = AutoModel.from_pretrained(model_path, **kwargs)
            else:
                raise
        _CURRENT_MODEL_PATH = model_path

        # If device auto-mapped to cpu but cuda is available, let user optionally move
        if DEVICE == "cuda":
            try:
                model = model.to("cuda")
            except Exception:
                # If model is already sharded across devices by accelerate, ignore
                pass

    # Always (re)build processor so min/max settings take effect
    pp_kwargs = {"trust_remote_code": True}
    if min_visual_tokens is not None:
        pp_kwargs["min_pixels"] = int(min_visual_tokens) * 28 * 28
    if max_visual_tokens is not None:
        pp_kwargs["max_pixels"] = int(max_visual_tokens) * 28 * 28
    processor = AutoProcessor.from_pretrained(model_path, **pp_kwargs)

    return processor, model


# ---- Media helpers ----
from PIL import Image


def _load_image(path: str, resized_height: Optional[int] = None, resized_width: Optional[int] = None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if resized_height and resized_width:
        im = im.resize((int(resized_width), int(resized_height)))
    return im


def _extract_video_frames(
    path: str,
    fps: Optional[float] = None,
    resized_height: Optional[int] = None,
    resized_width: Optional[int] = None,
    max_frames: Optional[int] = 64,
) -> List[Image.Image]:
    """Return a list of PIL.Image frames sampled from the video.

    Uses decord if available, else falls back to imageio.v3.
    """
    frames: List[Image.Image] = []
    try:
        import decord
        from decord import VideoReader

        decord.bridge.set_bridge("native")
        vr = VideoReader(path)
        total = len(vr)
        # Determine sampling
        if fps and fps > 0:
            try:
                native_fps = float(vr.get_avg_fps())
            except Exception:
                native_fps = None
            if native_fps and native_fps > 0:
                step = max(int(round(native_fps / fps)), 1)
                indices = list(range(0, total, step))
            else:
                # Fallback to uniform sampling
                num = min(max_frames or total, total)
                indices = [int(round(i * (total - 1) / max(1, num - 1))) for i in range(num)]
        else:
            # Uniform sampling to up to max_frames
            num = min(max_frames or total, total)
            indices = [int(round(i * (total - 1) / max(1, num - 1))) for i in range(num)]

        for idx in indices:
            arr = vr[idx].asnumpy()
            im = Image.fromarray(arr).convert("RGB")
            if resized_height and resized_width:
                im = im.resize((int(resized_width), int(resized_height)))
            frames.append(im)
    except Exception:
        # Fallback to imageio
        try:
            import imageio.v3 as iio

            # Read every frame; then sample based on fps/max_frames
            reader = iio.imiter(path)
            all_frames = [Image.fromarray(f).convert("RGB") for f in reader]
            total = len(all_frames)
            if total == 0:
                return []

            if fps and fps > 0:
                # No native FPS info; approximate with max_frames cap
                num = min(max_frames or total, total)
            else:
                num = min(max_frames or total, total)

            indices = [int(round(i * (total - 1) / max(1, num - 1))) for i in range(num)]
            for idx in indices:
                im = all_frames[idx]
                if resized_height and resized_width:
                    im = im.resize((int(resized_width), int(resized_height)))
                frames.append(im)
        except Exception:
            raise

    return frames


# ---- Prefer upstream keye_vl_utils.process_vision_info if available; else fallback ----
try:
    from keye_vl_utils import process_vision_info as _upstream_pvi  # type: ignore
except Exception:
    _upstream_pvi = None


def process_vision_info(messages, return_video_kwargs: bool = True):
    """Parse messages to produce (images, videos[, video_kwargs]).

    This function supports only local file paths coming from the Gradio UI.
    It returns:
      - images: List[Image.Image] for all images in order of appearance.
      - videos: List[List[Image.Image]] where each item is a list of frames for a video.
      - video_kwargs: currently an empty dict; reserved for model-specific args.
    """
    if _upstream_pvi is not None:
        # Delegate to upstream to handle more formats (URLs, base64, PIL, etc.)
        return _upstream_pvi(messages, return_video_kwargs=return_video_kwargs)
    images: List[Image.Image] = []
    # Each video will be a numpy array of shape (T, H, W, C)
    videos: List[object] = []
    video_kwargs = {}

    for turn in messages:
        for msg in turn:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            for item in content:
                t = item.get("type")
                if t == "image":
                    image_src = item.get("image")
                    # Expecting local file path from Gradio (e.g., /tmp/..., not file://)
                    if isinstance(image_src, str) and image_src.startswith("file://"):
                        image_src = image_src[len("file://") :]
                    rh = item.get("resized_height")
                    rw = item.get("resized_width")
                    if image_src and os.path.exists(image_src):
                        images.append(_load_image(image_src, rh, rw))
                elif t == "video":
                    video_src = item.get("video")
                    fps = item.get("fps")
                    rh = item.get("resized_height")
                    rw = item.get("resized_width")
                    mf = item.get("max_frames")
                    if isinstance(video_src, list):
                        # list of frame file paths
                        frames: List[Image.Image] = []
                        for p in video_src:
                            if isinstance(p, str) and p.startswith("file://"):
                                p = p[len("file://") :]
                            if p and os.path.exists(p):
                                frames.append(_load_image(p, rh, rw))
                        if frames:
                            try:
                                import numpy as np
                                arr = np.stack([np.asarray(f) for f in frames], axis=0)
                                videos.append(arr)
                            except Exception:
                                # As a last resort, keep list (may error downstream)
                                videos.append(frames)
                    else:
                        # assume single local path
                        if isinstance(video_src, str) and video_src.startswith("file://"):
                            video_src = video_src[len("file://") :]
                        if video_src and os.path.exists(video_src):
                            frames = _extract_video_frames(
                                video_src,
                                fps=fps,
                                resized_height=rh,
                                resized_width=rw,
                                max_frames=int(mf) if mf else 64,
                            )
                            if frames:
                                try:
                                    import numpy as np
                                    arr = np.stack([np.asarray(f) for f in frames], axis=0)
                                    videos.append(arr)
                                except Exception:
                                    videos.append(frames)

    if return_video_kwargs:
        return images, videos, video_kwargs
    return images, videos


# ---- Inference pipeline ----

def _apply_thinking_suffix(prompt: str, mode: str) -> str:
    # mode in {Auto, Think, No-Think}
    # normalize Chinese modes
    if mode in ("自动", "auto", "Auto"):
        mode = "Auto"
    elif mode in ("思考", "think", "Think"):
        mode = "Think"
    elif mode in ("不思考", "no-think", "no_think", "No-Think", "No_think"):
        mode = "No-Think"
    else:
        mode = "Auto"

    prompt = prompt or "Describe this image."
    if mode == "Think":
        if not prompt.endswith("/think"):
            prompt = f"{prompt}"
            if not prompt.endswith("."):
                prompt += "."
            prompt += "/think"
    elif mode == "No-Think":
        if not prompt.endswith("/no_think"):
            prompt = f"{prompt}"
            if not prompt.endswith("."):
                prompt += "."
            prompt += "/no_think"
    return prompt


def _build_messages_for_image(image_path: str, prompt: str, rh: Optional[int], rw: Optional[int], thinking_mode: str = "Auto"):
    image_uri = image_path if image_path.startswith("file://") else f"file://{image_path}"
    prompt = _apply_thinking_suffix(prompt, thinking_mode)
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri, **({} if not (rh and rw) else {"resized_height": int(rh), "resized_width": int(rw)})},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    ]


def _build_messages_for_video(
    video_path: str,
    prompt: str,
    fps: Optional[float],
    rh: Optional[int],
    rw: Optional[int],
    thinking_mode: str = "Auto",
    max_frames: Optional[int] = None,
):
    video_uri = video_path if video_path.startswith("file://") else f"file://{video_path}"
    prompt = _apply_thinking_suffix(prompt or "Describe this video.", thinking_mode)
    content = [
        {"type": "video", "video": video_uri},
        {"type": "text", "text": prompt},
    ]
    if fps and fps > 0:
        content[0]["fps"] = float(fps)
    if rh and rw:
        content[0]["resized_height"] = int(rh)
        content[0]["resized_width"] = int(rw)
    if max_frames and max_frames > 0:
        content[0]["max_frames"] = int(max_frames)

    return [[{"role": "user", "content": content}]]


def infer_image(
    image_path: str,
    prompt: str,
    model_path: str,
    max_new_tokens: int,
    use_flash_attn: bool,
    dtype_choice: Optional[str],
    resized_height: Optional[int],
    resized_width: Optional[int],
    thinking_mode: str,
    min_visual_tokens: Optional[float],
    max_visual_tokens: Optional[float],
):
    if not image_path:
        return "Please upload an image.", None

    try:
        proc, mdl = _load_model(
            model_path or MODEL_PATH_DEFAULT,
            use_flash_attn=use_flash_attn,
            min_visual_tokens=int(min_visual_tokens) if min_visual_tokens else None,
            max_visual_tokens=int(max_visual_tokens) if max_visual_tokens else None,
            dtype_choice=dtype_choice,
        )
        messages = _build_messages_for_image(image_path, prompt, resized_height, resized_width, thinking_mode)
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = proc(text=text, images=images or None, videos=videos or None, padding=True, return_tensors="pt", **video_kwargs)
        # Move to device if possible
        try:
            inputs = inputs.to("cuda")
        except Exception:
            pass

        generated_ids = mdl.generate(**inputs, max_new_tokens=int(max_new_tokens))
        # Trim input tokens to get only generated text
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        tok = getattr(proc, "tokenizer", None) or getattr(proc, "tokenizer_fast", None) or proc
        out_text = tok.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return out_text, text
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error: {e}\n\n{tb}", None


def infer_video(
    video_path: str,
    prompt: str,
    model_path: str,
    max_new_tokens: int,
    use_flash_attn: bool,
    dtype_choice: Optional[str],
    fps: Optional[float],
    resized_height: Optional[int],
    resized_width: Optional[int],
    thinking_mode: str,
    min_visual_tokens: Optional[float],
    max_visual_tokens: Optional[float],
    max_frames: Optional[int],
):
    if not video_path:
        return "Please upload a video.", None

    try:
        proc, mdl = _load_model(
            model_path or MODEL_PATH_DEFAULT,
            use_flash_attn=use_flash_attn,
            min_visual_tokens=int(min_visual_tokens) if min_visual_tokens else None,
            max_visual_tokens=int(max_visual_tokens) if max_visual_tokens else None,
            dtype_choice=dtype_choice,
        )
        messages = _build_messages_for_video(video_path, prompt, fps, resized_height, resized_width, thinking_mode, max_frames)
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = proc(text=text, images=images or None, videos=videos or None, padding=True, return_tensors="pt", **video_kwargs)
        try:
            inputs = inputs.to("cuda")
        except Exception:
            pass

        generated_ids = mdl.generate(**inputs, max_new_tokens=int(max_new_tokens))
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        tok = getattr(proc, "tokenizer", None) or getattr(proc, "tokenizer_fast", None) or proc
        out_text = tok.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return out_text, text
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error: {e}\n\n{tb}", None


# ---- Gradio UI ----
with gr.Blocks(title="Keye-VL Web 界面", css="footer {visibility: hidden}") as demo:
    gr.Markdown("""
    # Keye-VL Web 界面
    上传图片或视频并让模型进行描述。模型将在首次运行时加载。
    """)

    with gr.Row():
        model_path_inp = gr.Textbox(value=MODEL_PATH_DEFAULT, label="模型路径", show_label=True)
        max_new_tokens = gr.Slider(8, 16384, value=2048, step=8, label="最大新生成 Token")
        use_flash_attn_chk = gr.Checkbox(value=True, label="启用 Flash-Attn v2（若不可用则自动回退）")
        dtype_dd = gr.Dropdown(choices=["自动", "bf16", "fp16", "fp32"], value="自动", label="精度 (dtype)")

    with gr.Accordion("每张图像的视觉 Token（高级）", open=False):
        with gr.Row():
            min_tokens_inp = gr.Number(value=None, label="最小视觉 Token（如 256）")
            max_tokens_inp = gr.Number(value=None, label="最大视觉 Token（如 1280）")

    with gr.Tabs():
        with gr.TabItem("图片"):
            with gr.Row():
                image_inp = gr.Image(type="filepath", label="图片")
            prompt_img = gr.Textbox(value="请描述这张图片。", label="提示词")
            mode_img = gr.Dropdown(choices=["自动", "思考", "不思考"], value="自动", label="思维模式")
            with gr.Accordion("高级选项（可选）", open=False):
                rh_img = gr.Number(value=None, label="重设高度 (px)")
                rw_img = gr.Number(value=None, label="重设宽度 (px)")
            run_img = gr.Button("生成（图片）")
            out_img = gr.Textbox(label="输出")
            used_prompt_img = gr.Textbox(label="编码后的提示（调试）", visible=False)

            run_img.click(
                infer_image,
                    inputs=[image_inp, prompt_img, model_path_inp, max_new_tokens, use_flash_attn_chk, dtype_dd, rh_img, rw_img, mode_img, min_tokens_inp, max_tokens_inp],
                outputs=[out_img, used_prompt_img],
            )

        with gr.TabItem("视频"):
            with gr.Row():
                video_inp = gr.Video(label="视频")
            prompt_vid = gr.Textbox(value="请描述这个视频。", label="提示词")
            mode_vid = gr.Dropdown(choices=["自动", "思考", "不思考"], value="自动", label="思维模式")
            with gr.Accordion("高级选项（可选）", open=False):
                fps_inp = gr.Number(value=2.0, label="采样帧率 (FPS)")
                rh_vid = gr.Number(value=280, label="重设高度 (px)")
                rw_vid = gr.Number(value=280, label="重设宽度 (px)")
                max_frames_inp = gr.Slider(1, 128, value=32, step=1, label="最大帧数")
            run_vid = gr.Button("生成（视频）")
            out_vid = gr.Textbox(label="输出")
            used_prompt_vid = gr.Textbox(label="编码后的提示（调试）", visible=False)

            run_vid.click(
                infer_video,
                    inputs=[video_inp, prompt_vid, model_path_inp, max_new_tokens, use_flash_attn_chk, dtype_dd, fps_inp, rh_vid, rw_vid, mode_vid, min_tokens_inp, max_tokens_inp, max_frames_inp],
                outputs=[out_vid, used_prompt_vid],
            )

    gr.Markdown(
        """
        提示：
        - 可设置环境变量 VIDEO_MAX_PIXELS 限制视频视觉 token 上限，例如：`export VIDEO_MAX_PIXELS=25000000`。
        - 未安装 flash-attn 也可使用；勾选后会尝试启用，若不可用会自动回退。
        - 建议在带有 CUDA 的环境中使用以获得更快速度；仅 CPU 推理会较慢。
        """
    )


if __name__ == "__main__":
    # Respect GRADIO_SERVER_PORT/HOST if provided
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860)))
    server_host = os.environ.get("GRADIO_SERVER_HOST", "0.0.0.0")
    demo.queue().launch(server_name=server_host, server_port=server_port, show_error=True)
