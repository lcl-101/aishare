import os
import torch
from PIL import Image
import gradio as gr
from transformers import pipeline, BitsAndBytesConfig
import urllib.request

# 使用本地模型路径
MODEL_PATH = "checkpoints/medgemma-4b-it"
EXAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
EXAMPLE_IMAGE_PATH = "examples/Chest_Xray_PA_3-8-2010.png"

# 自动下载示例图片
os.makedirs("examples", exist_ok=True)
if not os.path.exists(EXAMPLE_IMAGE_PATH):
    try:
        urllib.request.urlretrieve(EXAMPLE_IMAGE_URL, EXAMPLE_IMAGE_PATH)
    except Exception as e:
        print(f"示例图片下载失败: {e}")

model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

# 加载 pipeline
pipe = pipeline(
    "image-text-to-text",
    model=MODEL_PATH,
    model_kwargs=model_kwargs,
)
pipe.model.generation_config.do_sample = False

def medgemma_infer(image, prompt):
    system_instruction = "You are an expert radiologist."
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]
    output = pipe(text=messages, max_new_tokens=300)
    response = output[0]["generated_text"][-1]["content"]
    return response

with gr.Blocks() as demo:
    gr.Markdown("# MedGemma WebUI Demo")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="上传医学图片")
            txt_input = gr.Textbox(label="输入问题", value="Describe this X-ray")
            btn = gr.Button("提交")
        with gr.Column():
            output = gr.Textbox(label="模型回复", lines=10)
    btn.click(fn=medgemma_infer, inputs=[img_input, txt_input], outputs=output)
    # 添加示例
    if os.path.exists(EXAMPLE_IMAGE_PATH):
        gr.Examples(
            examples=[[EXAMPLE_IMAGE_PATH, "Describe this X-ray"]],
            inputs=[img_input, txt_input],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
