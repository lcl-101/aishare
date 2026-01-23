import os
import json
import urllib.request
from typing import List, Optional

import gradio as gr
import torch
from PIL import Image, ImageDraw
from transformers import pipeline
from openai import OpenAI


APP_TITLE = "MedGemma 多模态演示"
EXAMPLES_DIR = "/workspace/medgemma/examples"
LOCAL_MODEL_DIR = os.environ.get(
    "MEDGEMMA_MODEL_DIR", "/workspace/medgemma/checkpoints/medgemma-1.5-4b-it"
)
HF_MODEL_ID = "google/medgemma-1.5-4b-it"

# LLM 翻译配置
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-4e3e00a0b4522d6d4c119ce2ddeb1722")
LLM_API_URL = os.environ.get("API_URL", "https://api.xxxxx.com/v1")
LLM_MODEL_NAME = os.environ.get("MODEL_NAME", "sykjtestuqwen2-5-72b-instruct")


EXAMPLE_FILES = {
    "xray": [
        (
            "chest_xray.png",
            "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        ),
    ],
    "longitudinal": [
        (
            "longitudinal_before.png",
            "https://storage.googleapis.com/hai-cd3-foundations-public-vault-entry/med_gemma/colab_example/cxr/longitudinal_cxr_before.png",
        ),
        (
            "longitudinal_after.png",
            "https://storage.googleapis.com/hai-cd3-foundations-public-vault-entry/med_gemma/colab_example/cxr/longitudinal_cxr_after.png",
        ),
    ],
}


def _create_placeholder(path: str, label: str) -> None:
    img = Image.new("RGB", (512, 512), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(16, 16), (496, 496)], outline="white", width=2)
    draw.text((32, 240), label, fill="white")
    img.save(path)


def ensure_examples() -> None:
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    for group in EXAMPLE_FILES.values():
        for filename, url in group:
            path = os.path.join(EXAMPLES_DIR, filename)
            if os.path.exists(path):
                print(f"[示例文件] 已存在: {filename}")
                continue
            try:
                print(f"[示例文件] 正在下载: {filename} ...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    with open(path, "wb") as f:
                        f.write(resp.read())
                print(f"[示例文件] 下载成功: {filename}")
            except Exception as e:
                print(f"[示例文件] 下载失败: {filename}, 错误: {e}, 创建占位图")
                _create_placeholder(path, filename)


def get_model_path() -> str:
    if os.path.isdir(LOCAL_MODEL_DIR):
        return LOCAL_MODEL_DIR
    return HF_MODEL_ID


_PIPE = None


def _select_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def get_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    dtype = _select_dtype()
    _PIPE = pipeline(
        "image-text-to-text",
        model=get_model_path(),
        model_kwargs={"device_map": "auto", "dtype": dtype},
    )
    _PIPE.model.generation_config.do_sample = False
    return _PIPE


def _extract_response(output) -> str:
    if isinstance(output, list) and output:
        generated = output[0].get("generated_text")
        if isinstance(generated, list) and generated:
            return generated[-1].get("content", "")
        if isinstance(generated, str):
            return generated
    return ""


def generate(messages, max_new_tokens: int = 400) -> str:
    pipe = get_pipe()
    output = pipe(
        text=messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return _extract_response(output)


def build_system_message(system_instruction: Optional[str]) -> List[dict]:
    if not system_instruction:
        return []
    return [{"role": "system", "content": [{"type": "text", "text": system_instruction}]}]


def run_text_only(prompt: str, system_instruction: str, max_new_tokens: int) -> tuple:
    messages = build_system_message(system_instruction)
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    result = generate(messages, max_new_tokens=max_new_tokens)
    translated = translate_to_chinese(result)
    return result, translated


def run_single_image(prompt: str, image: Image.Image, system_instruction: str, max_new_tokens: int) -> tuple:
    if image is None:
        return "请先上传一张图片。", ""
    messages = build_system_message(system_instruction)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    )
    result = generate(messages, max_new_tokens=max_new_tokens)
    translated = translate_to_chinese(result)
    return result, translated


def run_two_images(
    prompt: str,
    image1: Image.Image,
    image2: Image.Image,
    system_instruction: str,
    max_new_tokens: int,
) -> tuple:
    if image1 is None or image2 is None:
        return "请上传两张图片。", ""
    messages = build_system_message(system_instruction)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
            ],
        }
    )
    result = generate(messages, max_new_tokens=max_new_tokens)
    translated = translate_to_chinese(result)
    return result, translated


def _parse_bbox_response(text: str) -> List[dict]:
    start = text.find("```json")
    end = text.rfind("```")
    if start == -1 or end == -1 or end <= start:
        return []
    json_str = text[start + len("```json") : end].strip()
    try:
        return json.loads(json_str)
    except Exception:
        return []


def _draw_bboxes(image: Image.Image, boxes: List[dict]) -> Image.Image:
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    width, height = draw_image.size
    for item in boxes:
        box_2d = item.get("box_2d")
        label = item.get("label")
        if not box_2d or len(box_2d) != 4:
            continue
        y0, x0, y1, x1 = box_2d
        x0 = x0 / 1000 * width
        y0 = y0 / 1000 * height
        x1 = x1 / 1000 * width
        y1 = y1 / 1000 * height
        draw.rectangle([(x0, y0), (x1, y1)], outline="red", width=3)
        if label:
            draw.text((x0, max(0, y0 - 12)), label, fill="red")
    return draw_image


def run_localization(object_name: str, image: Image.Image, max_new_tokens: int):
    if image is None:
        return "请先上传一张图片。", "", None
    prompt = f"""Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.
Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.
Query:
Where is the {object_name}? Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]
    response = generate(messages, max_new_tokens=max_new_tokens)
    translated = translate_to_chinese(response)
    boxes = _parse_bbox_response(response)
    if not boxes:
        return response, translated, image
    return response, translated, _draw_bboxes(image, boxes)


def get_example_paths(key: str) -> List[str]:
    return [os.path.join(EXAMPLES_DIR, name) for name, _ in EXAMPLE_FILES[key]]


def translate_to_chinese(text: str) -> str:
    """使用 LLM 将英文翻译为中文"""
    if not text or not text.strip():
        return "没有内容需要翻译。"
    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的医学翻译助手。请将以下英文医学内容准确翻译为中文，保持专业术语的准确性。只输出翻译结果，不要添加任何解释。",
                },
                {"role": "user", "content": text},
            ],
            max_tokens=4096,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"翻译失败: {e}"


def build_app() -> gr.Blocks:
    ensure_examples()
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            """# AI 技术分享频道  
[https://www.youtube.com/@rongyikanshijie-ai](https://www.youtube.com/@rongyikanshijie-ai)"""
        )
        gr.Markdown("使用本地模型与示例文件进行多模态演示。界面为中文，示例提示词保持原文。")

        with gr.Tab("文本问答"):
            prompt = gr.Textbox(
                label="问题（示例保持原文）",
                value="How do you differentiate bacterial from viral pneumonia?",
            )
            system_instruction = gr.Textbox(
                label="系统指令",
                value="You are a helpful medical assistant.",
            )
            max_tokens = gr.Slider(
                64, 2000, value=500, step=1, label="最大输出长度"
            )
            run_btn = gr.Button("生成")
            with gr.Row():
                output = gr.Textbox(label="原始输出", lines=10)
                translated_output = gr.Textbox(label="中文翻译", lines=10)
            run_btn.click(run_text_only, [prompt, system_instruction, max_tokens], [output, translated_output])

        with gr.Tab("图文推理"):
            image = gr.Image(type="pil", label="上传图片")
            prompt = gr.Textbox(
                label="提示词（示例保持原文）",
                value="Describe this X-ray",
            )
            system_instruction = gr.Textbox(
                label="系统指令",
                value="You are an expert radiologist.",
            )
            max_tokens = gr.Slider(64, 2000, value=300, step=1, label="最大输出长度")
            run_btn = gr.Button("生成")
            with gr.Row():
                output = gr.Textbox(label="原始输出", lines=10)
                translated_output = gr.Textbox(label="中文翻译", lines=10)
            run_btn.click(run_single_image, [prompt, image, system_instruction, max_tokens], [output, translated_output])
            gr.Examples(
                examples=[[get_example_paths("xray")[0], "Describe this X-ray", "You are an expert radiologist.", 300]],
                inputs=[image, prompt, system_instruction, max_tokens],
                label="示例",
            )

        with gr.Tab("胸片解剖定位"):
            image = gr.Image(type="pil", label="上传图片")
            object_name = gr.Textbox(label="目标结构（示例保持原文）", value="right clavicle")
            max_tokens = gr.Slider(64, 2000, value=800, step=1, label="最大输出长度")
            run_btn = gr.Button("生成")
            with gr.Row():
                output_text = gr.Textbox(label="原始输出", lines=10)
                translated_output = gr.Textbox(label="中文翻译", lines=10)
            output_image = gr.Image(label="定位结果")
            run_btn.click(
                run_localization,
                [object_name, image, max_tokens],
                [output_text, translated_output, output_image],
            )
            gr.Examples(
                examples=[[get_example_paths("xray")[0], "right clavicle", 800]],
                inputs=[image, object_name, max_tokens],
                label="示例",
            )

        with gr.Tab("纵向对比"):
            image1 = gr.Image(type="pil", label="上传图片（前）")
            image2 = gr.Image(type="pil", label="上传图片（后）")
            prompt = gr.Textbox(
                label="提示词（示例保持原文）",
                value=(
                    "Provide a comparison of these two images and include details from\n"
                    "the image which students should take note of when reading longitudinal CXR"
                ),
            )
            system_instruction = gr.Textbox(
                label="系统指令",
                value="You are an expert radiologist.",
            )
            max_tokens = gr.Slider(64, 2000, value=400, step=1, label="最大输出长度")
            run_btn = gr.Button("生成")
            with gr.Row():
                output = gr.Textbox(label="原始输出", lines=10)
                translated_output = gr.Textbox(label="中文翻译", lines=10)
            run_btn.click(
                run_two_images,
                [prompt, image1, image2, system_instruction, max_tokens],
                [output, translated_output],
            )
            gr.Examples(
                examples=[
                    [
                        get_example_paths("longitudinal")[0],
                        get_example_paths("longitudinal")[1],
                        (
                            "Provide a comparison of these two images and include details from\n"
                            "the image which students should take note of when reading longitudinal CXR"
                        ),
                        "You are an expert radiologist.",
                        400,
                    ]
                ],
                inputs=[image1, image2, prompt, system_instruction, max_tokens],
                label="示例",
            )
    return demo


if __name__ == "__main__":
    ensure_examples()
    app = build_app()
    app.launch(server_name="0.0.0.0")