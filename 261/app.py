from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)

from inference import WHISPER_FEAT_CFG, build_prompt, prepare_inputs

DEFAULT_CHECKPOINT = "checkpoints/GLM-ASR-Nano-2512"
EXAMPLE_FILES = [
    ["examples/example_en.wav"],
    ["examples/example_zh.wav"],
]
MAX_NEW_TOKENS_LIMIT = 8192  # UI cap; actual usable length depends on context budget


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(device: str):
    return torch.bfloat16 if device != "cpu" else torch.float32


@lru_cache(maxsize=1)
def load_resources(checkpoint_dir: str, device: str):
    checkpoint_dir = str(Path(checkpoint_dir).expanduser())
    tokenizer_source = checkpoint_dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=_select_dtype(device),
        trust_remote_code=True,
    ).to(device)
    model.eval()

    return model, tokenizer, feature_extractor, config.merge_factor


# Load model once at startup so the UI is ready to use immediately.
DEFAULT_DEVICE = get_default_device()
MODEL, TOKENIZER, FEATURE_EXTRACTOR, MERGE_FACTOR = load_resources(
    DEFAULT_CHECKPOINT, DEFAULT_DEVICE
)


def transcribe_audio(
    audio_path: Optional[str],
    max_new_tokens: int,
):
    if not audio_path:
        return "Please provide an audio file."

    try:
        batch = build_prompt(
            Path(audio_path), TOKENIZER, FEATURE_EXTRACTOR, MERGE_FACTOR
        )
        model_inputs, prompt_len = prepare_inputs(batch, torch.device(DEFAULT_DEVICE))

        with torch.inference_mode():
            generated = MODEL.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        transcript_ids = generated[0, prompt_len:].cpu().tolist()
        transcript = TOKENIZER.decode(
            transcript_ids, skip_special_tokens=True
        ).strip()
        return transcript or "[Empty transcription]"
    except Exception as exc:  # noqa: BLE001
        return f"Error during transcription: {exc}"


def build_interface():
    with gr.Blocks(title="GLM-ASR Web Demo") as demo:
        gr.Markdown(
            """
            # GLM-ASR Gradio Demo
            - 程序启动时已加载本地模型 `checkpoints/GLM-ASR-Nano-2512`。
            - 选择或上传音频，点击 **Transcribe** 获取转写结果。
            - `Max new tokens` 上限 8192；数值越大越占显存/内存且更慢，建议按需调小。
            """
        )

        max_tokens = gr.Slider(
            minimum=16,
            maximum=MAX_NEW_TOKENS_LIMIT,
            value=128,
            step=1,
            label="Max new tokens",
        )

        audio_input = gr.Audio(
            label="Audio",
            sources=["upload", "microphone"],
            type="filepath",
        )
        output_box = gr.Textbox(label="Transcript", lines=6)

        transcribe_btn = gr.Button("Transcribe")
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, max_tokens],
            outputs=output_box,
        )

        gr.Examples(
            label="Examples",
            examples=EXAMPLE_FILES,
            inputs=audio_input,
            examples_per_page=10,
            cache_examples=False,
        )

    return demo


def main():
    demo = build_interface()
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
