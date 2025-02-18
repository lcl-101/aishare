import torch
from diffusers import FluxPipeline
import gradio as gr

# 修改加载模型路径为本地的 checkpoints/FLUX.1-dev，并将 torch_dtype 改为 float16 以降低显存占用
pipe = FluxPipeline.from_pretrained("checkpoints/FLUX.1-dev", torch_dtype=torch.bfloat16)

# 加载 LoRA 权重，路径为本地 checkpoints/how2draw 下的 lora 模型
pipe.load_lora_weights("checkpoints/how2draw/")

pipe.enable_model_cpu_offload()

def generate_image(prompt):
    image = pipe(
        prompt,
        height=512,
        width=768,
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("## Flux 图像生成 WebUI")
    # 图片预览区域
    image_preview = gr.Image(type="pil")
    # 将用户提示词输入层改为下拉菜单，选项通过分隔符 "；" 分开
    prompt_options = [
        "an owl, How3Draw",
        "a dolphin, How2Draw",
        "A silhouette of a girl performing a ballet pose, with elegant lines to suggest grace and movement. The background can include simple outlines of ballet shoes and a music note. The image should convey elegance and poise in a minimalistic style, How2Draw",
        "Yorkshire Terrier with smile, How2Draw",
        "a druid, How2Draw",
        "sunflower, How2Draw"
    ]
    prompt_input = gr.Dropdown(label="提示词", choices=prompt_options, value=prompt_options[0])
    # 推理按钮
    infer_button = gr.Button("推理")
    
    infer_button.click(fn=generate_image, inputs=prompt_input, outputs=image_preview)

demo.launch(server_name='0.0.0.0')
