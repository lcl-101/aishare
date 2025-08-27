"""Simple Gradio demo for Wan S2V 14B (speech/image to video).

This app focuses on two common example scenarios and exposes a minimal
set of generation parameters. It loads the S2V model once on startup
and reuses it across requests.

Run:  python app.py  (then open the printed Gradio URL)
"""

import os
import time
import gradio as gr
import torch
from datetime import datetime

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video, merge_video_audio

# -------------------- Global / Lazy Model -------------------- #

# Hardcoded checkpoint directory (user requested removing env var logic)
CKPT_DIR = "checkpoints/Wan2.2-S2V-14B"
TASK = "s2v-14B"
CFG = WAN_CONFIGS[TASK]

_model = None  # type: wan.WanS2V | None


def load_model(convert_model_dtype: bool = False, offload_model: bool = True):
    global _model
    if _model is None:
        _model = wan.WanS2V(
            config=CFG,
            checkpoint_dir=CKPT_DIR,
            device_id=0,
            rank=0,
            convert_model_dtype=convert_model_dtype,
        )
    return _model


# -------------------- Generation -------------------- #

def generate_video(prompt: str,
                   ref_image: str,
                   audio_file: str,
                   pose_video: str | None,
                   size: str,
                   infer_frames: int,
                   guide_scale: float,
                   sampling_steps: int,
                   shift: float,
                   seed: int,
                   start_from_ref: bool,
                   offload_model: bool):
    if not prompt:
        return None, "Prompt is empty"
    if not ref_image:
        return None, "Reference image is required"
    if not audio_file:
        return None, "Audio file is required"

    # 现在固定不再转换 dtype（已在权重中 bfloat16），也不支持引用首帧模式开关
    model = load_model(convert_model_dtype=False, offload_model=offload_model)

    # random seed
    if seed is None or seed < 0:
        seed = int(time.time() * 1000) % 2**31

    torch.manual_seed(seed)

    max_area = MAX_AREA_CONFIGS[size]

    video_tensor = model.generate(
        input_prompt=prompt,
        ref_image_path=ref_image,
        audio_path=audio_file,
        num_repeat=None,  # let internal logic decide based on audio length
        pose_video=pose_video if pose_video else None,
        max_area=max_area,
        infer_frames=infer_frames,
        shift=shift,
        sample_solver='unipc',
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
    init_first_frame=start_from_ref,
    )

    # Save + merge audio
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:40]
    out_name = f"{TASK}_{size}_{safe_prompt}_{stamp}.mp4"
    save_video(
        tensor=video_tensor[None],
        save_file=out_name,
        fps=CFG.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    merge_video_audio(video_path=out_name, audio_path=audio_file)

    return out_name, f"Done. Seed={seed}. Saved to {out_name}"


# -------------------- UI -------------------- #

EXAMPLE_1_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, "
    "and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the "
    "sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing "
    "atmosphere of the seaside.")

examples = [
    [
        EXAMPLE_1_PROMPT,
        "examples/i2v_input.JPG",
        "examples/talk.wav",
        None,
    ],
    [
        "a person is singing",
        "examples/pose.png",
        "examples/sing.MP3",
        "examples/pose.mp4",
    ],
]

with gr.Blocks(title="Wan S2V 14B 演示") as demo:
    gr.Markdown("## Wan S2V 14B 演示\n上传或选择示例，生成语音驱动视频。")
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="文本提示", lines=5, value=EXAMPLE_1_PROMPT)
            ref_image = gr.Image(label="参考图像", type="filepath")
            audio = gr.Audio(label="音频 (wav/mp3)", type="filepath")
            pose = gr.Video(label="姿态视频(可选)")
        with gr.Column():
            size = gr.Dropdown(
                choices=list(MAX_AREA_CONFIGS.keys()),
                value="1024*704",
                label="分辨率 (高*宽)")
            infer_frames = gr.Slider(
                minimum=16,
                maximum=160,
                step=4,
                value=80,
                label="每段帧数")
            guide_scale = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                step=0.1,
                value=CFG.sample_guide_scale,
                label="引导系数")
            sampling_steps = gr.Slider(
                minimum=10,
                maximum=80,
                step=1,
                value=CFG.sample_steps,
                label="采样步数")
            shift = gr.Slider(
                minimum=1.0,
                maximum=8.0,
                step=0.1,
                value=CFG.sample_shift,
                label="噪声移位 shift")
            seed = gr.Number(value=-1, label="随机种子 (-1 随机)")
            start_from_ref = gr.Checkbox(value=False, label="使用参考图像作为首帧")
            offload_model = gr.Checkbox(
                value=False, label="模型卸载以省显存 (更慢)")

    run_btn = gr.Button("开始生成", variant="primary")
    with gr.Row():
        output_video = gr.Video(label="输出视频")
        status = gr.Textbox(label="状态", interactive=False)

    gr.Examples(
        examples=examples,
        inputs=[prompt, ref_image, audio, pose],
        label="示例",
    )

    run_btn.click(
        fn=generate_video,
        inputs=[prompt, ref_image, audio, pose, size, infer_frames,
                guide_scale, sampling_steps, shift, seed, start_from_ref, offload_model],
        outputs=[output_video, status],
        api_name="generate",
    )

    gr.Markdown("模型首次生成时加载。高分辨率与更多帧会显著增加时间与显存。")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
