import os
import requests
import torch
import gradio as gr
from PIL import Image
from diffusers import DiffusionPipeline

MODEL_DIR = "/workspace/outfit/checkpoints/Qwen-Image-Edit-2511"
LORA_DIR = "/workspace/outfit/checkpoints/QIE-2511-Extract-Outfit"
SAMPLE_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
SAMPLE_IMAGE_PATH = "/workspace/outfit/sample.png"


def _download_sample_image() -> bool:
    """下载示例图片，成功返回 True，失败返回 False。"""
    if os.path.exists(SAMPLE_IMAGE_PATH):
        return True
    try:
        os.makedirs(os.path.dirname(SAMPLE_IMAGE_PATH), exist_ok=True)
        resp = requests.get(SAMPLE_IMAGE_URL, timeout=30)
        resp.raise_for_status()
        with open(SAMPLE_IMAGE_PATH, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"[警告] 示例图片下载失败，跳过: {e}")
        return False


def _load_pipeline() -> DiffusionPipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
    )
    if device == "cuda":
        pipe.to(device)
    pipe.load_lora_weights(LORA_DIR)
    return pipe


HAS_SAMPLE = _download_sample_image()
PIPE = _load_pipeline()


def generate_image(input_image: Image.Image, prompt: str) -> Image.Image:
    if input_image is None:
        raise gr.Error("请先上传或选择一张图片。")
    if not prompt or not prompt.strip():
        raise gr.Error("请输入提示词。")
    result = PIPE(image=input_image, prompt=prompt).images[0]
    return result


with gr.Blocks(title="Qwen Image Edit") as demo:
    gr.Markdown(
        """
# AI 技术分享频道  
[https://www.youtube.com/@rongyi-ai](https://www.youtube.com/@rongyi-ai)
"""
    )

    gr.Markdown("## Qwen 图像编辑（本地模型）")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="输入图片", type="pil")
            prompt = gr.Textbox(
                label="提示词",
                value="Extract the clothing and create a flat mockup.",
                lines=2,
            )
            run_btn = gr.Button("生成")
        with gr.Column():
            output_image = gr.Image(label="输出图片", type="pil")

    if HAS_SAMPLE:
        gr.Examples(
            examples=[[SAMPLE_IMAGE_PATH, "Extract the clothing and create a flat mockup."]],
            inputs=[input_image, prompt],
            label="示例",
        )

    run_btn.click(
        fn=generate_image,
        inputs=[input_image, prompt],
        outputs=[output_image],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
