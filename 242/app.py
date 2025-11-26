import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import gradio as gr
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "checkpoints" / "HunyuanOCR"
MODEL_PATH = Path(os.environ.get("HUNYUAN_OCR_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
DEFAULT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，"
    "文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)
MAX_NEW_TOKENS = 1024
EXAMPLE_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/chat-ui/tools-dark.png"
EXAMPLE_IMAGE_PATH = BASE_DIR / "assets" / "examples" / "tools-dark.png"


def clean_repeated_substrings(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """Clean repeated substrings in the decoded text output."""
    if isinstance(text, list):
        return [clean_repeated_substrings(t) for t in text]

    n = len(text)
    if n < 8000:
        return text

    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[: n - length * (count - 1)]

    return text


def _ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


def ensure_example_image(path: Path = EXAMPLE_IMAGE_PATH, url: str = EXAMPLE_IMAGE_URL) -> Optional[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        path.write_bytes(response.content)
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"示例图片下载失败: {exc}")
        return None


@lru_cache(maxsize=1)
def load_pipeline(model_dir: str = str(MODEL_PATH)):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型目录 {model_path} 不存在，请检查 checkpoints/HunyuanOCR 是否可用或设置 HUNYUAN_OCR_MODEL_PATH 环境变量"
        )

    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return processor, model


def build_examples(default_prompt: str = DEFAULT_PROMPT) -> List[List[str]]:
    example_path = ensure_example_image()
    if example_path is None:
        return []
    return [[str(example_path), default_prompt]]


def _extract_input_ids(batch_encoding):
    if hasattr(batch_encoding, "input_ids") and batch_encoding.input_ids is not None:
        return batch_encoding.input_ids
    if "input_ids" in batch_encoding:
        return batch_encoding["input_ids"]
    if "inputs" in batch_encoding:
        return batch_encoding["inputs"]
    raise ValueError("无法在输入张量中找到 input_ids")


def run_ocr(image: Optional[Image.Image], prompt: str) -> str:
    if image is None:
        raise gr.Error("请上传待识别的图片")

    prompt_text = prompt.strip() or DEFAULT_PROMPT
    image = _ensure_rgb(image)
    processor, model = load_pipeline()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    input_ids = _extract_input_ids(inputs)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]

    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    cleaned = clean_repeated_substrings(decoded)
    return cleaned[0] if isinstance(cleaned, list) else cleaned


def build_interface() -> gr.Blocks:
    examples = build_examples()
    with gr.Blocks(title="Hunyuan OCR") as demo:
        gr.Markdown("""# Hunyuan OCR WebUI\n上传文档图片后点击 **识别** 获取 Markdown 结果。""")

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                image_component = gr.Image(label="上传图片", type="pil", height=380)
            with gr.Column(scale=2):
                prompt_component = gr.Textbox(
                    label="识别提示词",
                    value=DEFAULT_PROMPT,
                    lines=6,
                    max_lines=8,
                )
                run_button = gr.Button("识别", variant="primary", scale=0)

        if examples:
            gr.Examples(
                label="示例",
                examples=examples,
                inputs=[image_component, prompt_component],
                cache_examples=False,
            )

        output_component = gr.Markdown(label="识别结果")

        run_button.click(
            run_ocr,
            inputs=[image_component, prompt_component],
            outputs=output_component,
        )

    return demo


def main():
    load_pipeline()
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
