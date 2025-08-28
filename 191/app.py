import os
import tempfile
import gradio as gr
import torch
import torchaudio
from loguru import logger
from typing import Optional, Tuple
import random
import numpy as np

from hunyuanvideo_foley.utils.model_utils import load_model
from hunyuanvideo_foley.utils.feature_utils import feature_process
from hunyuanvideo_foley.utils.model_utils import denoise_process
from hunyuanvideo_foley.utils.media_utils import merge_audio_video

"""Minimal dark-mode Gradio app (app.py)

This file replicates the functionality of gradio_app.py but applies a cleaner,
fully dark theme with fewer gradients and visual distractions.

Run:
  python app.py

Optional:
  python app.py --model_path checkpoints/HunyuanVideo-Foley --port 7860 --host 0.0.0.0
"""

# -------------------- Model Path Resolution --------------------
REQUIRED_MODEL_FILES = [
    "hunyuanvideo_foley.pth",
    "vae_128d_48k.pth",
    "synchformer_state_dict.pth",
]


def _candidate_model_dirs():
    return [
        "checkpoints/HunyuanVideo-Foley",
        "./HunyuanVideo-Foley",
        "./checkpoints",
        "./pretrained_models",
        "./pretrained",
        os.getcwd(),
    ]


def _is_valid_model_dir(path: str) -> bool:
    return os.path.isdir(path) and all(os.path.isfile(os.path.join(path, f)) for f in REQUIRED_MODEL_FILES)


def resolve_model_path(cli_arg: Optional[str] = None) -> str:
    if cli_arg and _is_valid_model_dir(cli_arg):
        return cli_arg
    env_path = os.environ.get("HIFI_FOLEY_MODEL_PATH")
    if env_path and _is_valid_model_dir(env_path):
        return env_path
    for cand in _candidate_model_dirs():
        if _is_valid_model_dir(cand):
            return cand
    return cli_arg or env_path or "./checkpoints/HunyuanVideo-Foley"

CONFIG_PATH = "configs/hunyuanvideo-foley-xxl.yaml"

# -------------------- Globals --------------------
model_dict = None
cfg = None
device = None
MODEL_PATH = None

# -------------------- Core Helpers --------------------

def setup_device(device_str: str = "auto", gpu_id: int = 0) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            dev = torch.device(f"cuda:{gpu_id}")
            logger.info(f"Using CUDA device: {dev}")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
            logger.info("Using MPS device")
        else:
            dev = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        dev = torch.device(f"cuda:{gpu_id}" if device_str == "cuda" else device_str)
        logger.info(f"Using specified device: {dev}")
    return dev


def auto_load_models() -> str:
    global model_dict, cfg, device
    try:
        if not os.path.exists(MODEL_PATH):
            return f"❌ Model path not found: {MODEL_PATH}"
        if not os.path.exists(CONFIG_PATH):
            return f"❌ Config file not found: {CONFIG_PATH}"
        device = setup_device("auto", 0)
        logger.info("Loading model...")
        model_dict, cfg = load_model(MODEL_PATH, CONFIG_PATH, device)
        return "✅ Model loaded successfully"
    except Exception as e:
        logger.exception("Model loading failed")
        return f"❌ Model loading failed: {e}".strip()


def infer_single_video(
    video_file,
    text_prompt: str,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 50,
    sample_nums: int = 1,
) -> Tuple[list, str]:
    global model_dict, cfg, device
    if model_dict is None:
        return [], "❌ Load model first"
    if video_file is None:
        return [], "❌ Please upload a video file"
    text_prompt = (text_prompt or "").strip()
    try:
        visual_feats, text_feats, audio_len_in_s = feature_process(video_file, text_prompt, model_dict, cfg)
        audio, sample_rate = denoise_process(
            visual_feats,
            text_feats,
            audio_len_in_s,
            model_dict,
            cfg,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            batch_size=sample_nums,
        )
        temp_dir = tempfile.mkdtemp()
        outputs = []
        for i in range(sample_nums):
            wav_path = os.path.join(temp_dir, f"gen_{i+1}.wav")
            torchaudio.save(wav_path, audio[i], sample_rate)
            video_out = os.path.join(temp_dir, f"result_{i+1}.mp4")
            merge_audio_video(wav_path, video_file, video_out)
            outputs.append(video_out)
        return outputs, f"✅ Generated {sample_nums} sample(s)"
    except Exception as e:
        logger.exception("Inference failed")
        return [], f"❌ Inference failed: {e}".strip()

# -------------------- UI --------------------

def create_interface():
    dark_css = """
    body, .gradio-container {background:#0d0f12 !important; color:#d1d5db !important; font-family:'Inter',system-ui,sans-serif;}
    .gradio-container * {box-shadow:none !important;}
    .block, .gr-panel, .gr-box, .form, .tabs, .tabitem {background:#161b21 !important; border:1px solid #232a31 !important; border-radius:8px !important;}
    .gr-button {background:#2b3340 !important; border:1px solid #3a4552 !important; color:#e5e7eb !important; border-radius:6px !important; font-weight:500;}
    .gr-button.primary, .generate-btn {background:#4556d8 !important; border:1px solid #4f63ef !important;}
    .gr-button.primary:hover, .generate-btn:hover {background:#5364ea !important;}
    .gr-button:hover {background:#343e4c !important;}
    textarea, input, .gr-text-input textarea {background:#101419 !important; border:1px solid #232a31 !important; color:#e5e7eb !important;}
    label, .gradio-container h1, .gradio-container h2, .gradio-container h3 {color:#e5e7eb !important; font-weight:500;}
    p, .markdown {color:#9ca3af !important;}
    .header-bar {background:#161b21 !important; border:1px solid #232a31 !important; padding:0.9rem 1rem 0.75rem; border-radius:10px;}
    .header-bar h1 {margin:0; font-size:1.4rem; font-weight:600; color:#f1f5f9;}
    .compact-row {gap:0.9rem;}
    .examples-grid {display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:0.75rem;}
    .example-item {background:#161b21; border:1px solid #232a31; padding:0.55rem 0.6rem 0.7rem; border-radius:8px;}
    .example-item video {height:105px !important; object-fit:cover; background:#000; border-radius:4px;}
    .example-caption {font-size:0.7rem; line-height:1.05rem; min-height:2.1rem; color:#c3c9d1;}
    .footer {text-align:center; font-size:11px; color:#556270; margin-top:1.25rem; padding:0.75rem 0 0.5rem;}
    .status-box textarea {background:#101419 !important;}
    .gr-accordion {background:#161b21 !important; border:1px solid #232a31 !important;}
    .gr-accordion .label-wrap {background:#161b21 !important;}
    video {background:#000 !important;}
    .no-border {border:none !important;}
    .minimal-border {border:1px solid #232a31 !important;}
    """

    with gr.Blocks(css=dark_css, title="HunyuanVideo-Foley (Dark)") as demo:
        # Header
        with gr.Row(elem_classes=["header-bar"]):
            gr.Markdown("""# HunyuanVideo-Foley\n**Text/Video → Audio generation**""")
        # Main Row
        with gr.Row(elem_classes=["compact-row"]):
            with gr.Column(scale=1):
                video_input = gr.Video(label="Video", height=300)
                text_input = gr.Textbox(label="Prompt (optional)", placeholder="Describe target audio", lines=2)
                with gr.Row():
                    guidance_scale = gr.Slider(1.0, 10.0, value=4.5, step=0.1, label="CFG")
                    inference_steps = gr.Slider(10, 100, value=50, step=5, label="Steps")
                    sample_nums = gr.Slider(1, 6, value=1, step=1, label="Samples")
                generate_btn = gr.Button("Generate", elem_classes=["generate-btn"], variant="primary")
            with gr.Column(scale=1):
                video_output_1 = gr.Video(label="Sample 1", height=250)
                with gr.Row():
                    video_output_2 = gr.Video(visible=False, height=120)
                    video_output_3 = gr.Video(visible=False, height=120)
                with gr.Row():
                    video_output_4 = gr.Video(visible=False, height=120)
                    video_output_5 = gr.Video(visible=False, height=120)
                video_output_6 = gr.Video(visible=False, height=120)
                result_text = gr.Textbox(label="Status", interactive=False, lines=2, elem_classes=["status-box"])
        # Examples
        with gr.Accordion("Examples", open=False):
            examples_data = [
                {"caption": "A person walks on frozen ice", "video_path": "examples/1_video.mp4", "result_path": "examples/1_result.mp4"},
                {"caption": "With a faint sound as their hands parted, the two embraced", "video_path": "examples/2_video.mp4", "result_path": "examples/2_result.mp4"},
                {"caption": "Light bouncing footsteps like glass marbles", "video_path": "examples/3_video.mp4", "result_path": "examples/3_result.mp4"},
                {"caption": "Gentle stream with serene piano", "video_path": "examples/4_video.mp4", "result_path": "examples/4_result.mp4"},
                {"caption": "Snow crunching under snowboard", "video_path": "examples/5_video.mp4", "result_path": "examples/5_result.mp4"},
                {"caption": "Crackling fire and popping leaves", "video_path": "examples/6_video.mp4", "result_path": "examples/6_result.mp4"},
                {"caption": "Scooter engine humming accelerating", "video_path": "examples/7_video.mp4", "result_path": "examples/7_result.mp4"},
                {"caption": "Water splash and loud thud", "video_path": "examples/8_video.mp4", "result_path": "examples/8_result.mp4"},
            ]
            example_buttons = []
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Click to load example video & prompt.")
            with gr.Row():
                with gr.Column():
                    with gr.Row(elem_classes=["examples-grid"]):
                        for ex in examples_data:
                            with gr.Column(elem_classes=["example-item"]):
                                if os.path.exists(ex["video_path"]):
                                    gr.Video(value=ex["video_path"], interactive=False, show_label=False)
                                else:
                                    gr.HTML(f"<div style='height:110px;display:flex;align-items:center;justify-content:center;font-size:11px;color:#64748b;border:1px dashed #334155;border-radius:6px;'>Missing</div>")
                                gr.Markdown(f"<div class='example-caption'>{ex['caption'][:40]}{'...' if len(ex['caption'])>40 else ''}</div>")
                                btn = gr.Button("Load", size="sm")
                                example_buttons.append((btn, ex))

        def process(video_file, text_prompt, guidance_scale, steps, sample_n):
            vids, status = infer_single_video(video_file, text_prompt, guidance_scale, steps, int(sample_n))
            # pad to 6
            vids += [None] * (6 - len(vids))
            return (
                vids[0], vids[1], vids[2], vids[3], vids[4], vids[5], status
            )

        def update_visibility(sample_n):
            n = int(sample_n)
            return [
                gr.update(visible=True),
                gr.update(visible=n >= 2),
                gr.update(visible=n >= 3),
                gr.update(visible=n >= 4),
                gr.update(visible=n >= 5),
                gr.update(visible=n >= 6),
            ]

        sample_nums.change(
            fn=update_visibility,
            inputs=[sample_nums],
            outputs=[video_output_1, video_output_2, video_output_3, video_output_4, video_output_5, video_output_6],
        )

        generate_btn.click(
            fn=process,
            inputs=[video_input, text_input, guidance_scale, inference_steps, sample_nums],
            outputs=[video_output_1, video_output_2, video_output_3, video_output_4, video_output_5, video_output_6, result_text],
        )

        for btn, ex in example_buttons:
            def _mk_handler(e):
                def _h():
                    vfile = e['video_path'] if os.path.exists(e['video_path']) else None
                    return vfile, e['caption'], e['result_path'] if os.path.exists(e['result_path']) else None, f"Loaded example: {e['caption'][:45]}" + ("..." if len(e['caption'])>45 else "")
                return _h
            btn.click(fn=_mk_handler(ex), outputs=[video_input, text_input, video_output_1, result_text])

        gr.HTML("""<div class='footer'>HunyuanVideo-Foley · Dark UI</div>""")
    return demo

# -------------------- Seed --------------------

def set_manual_seed(global_seed: int):
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

# -------------------- Main --------------------
if __name__ == "__main__":
    import argparse
    set_manual_seed(1)
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO")

    parser = argparse.ArgumentParser(description="Dark UI for HunyuanVideo-Foley")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    MODEL_PATH = resolve_model_path(args.model_path)
    if not _is_valid_model_dir(MODEL_PATH):
        logger.warning(f"Model directory incomplete: {MODEL_PATH}")
        miss = [f for f in REQUIRED_MODEL_FILES if not os.path.isfile(os.path.join(MODEL_PATH, f))]
        logger.warning(f"Missing: {miss}")

    logger.info(f"Using model dir: {MODEL_PATH}")
    model_status = auto_load_models()
    logger.info(model_status)

    app = create_interface()
    if "✅" in model_status:
        logger.info("Interface ready")

    app.launch(server_name=args.host, server_port=args.port, share=False, debug=False, show_error=True)
