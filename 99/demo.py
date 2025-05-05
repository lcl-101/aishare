import torch
import gradio as gr
from f_lite import FLitePipeline
import threading
import gc

# Trick required because it is not a native diffusers model
from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES, ALL_IMPORTABLE_CLASSES
LOADABLE_CLASSES["f_lite"] = LOADABLE_CLASSES["f_lite.model"] = {"DiT": ["save_pretrained", "from_pretrained"]}
ALL_IMPORTABLE_CLASSES["DiT"] = ["save_pretrained", "from_pretrained"]

# 支持多模型切换，使用全局变量和锁
pipeline_lock = threading.Lock()
pipeline = None
current_model_path = None

def load_pipeline(model_path):
    global pipeline, current_model_path
    with pipeline_lock:
        if pipeline is not None and current_model_path != model_path:
            # 释放旧 pipeline
            pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if pipeline is None or current_model_path != model_path:
            pipeline = FLitePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            pipeline.enable_model_cpu_offload()
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
            current_model_path = model_path

def generate_image(
    prompt,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    negative_prompt,
    model_path
):
    load_pipeline(model_path)
    # 再次确认 pipeline 已加载正确模型
    if current_model_path != model_path or pipeline is None:
        raise RuntimeError("Pipeline not loaded correctly.")
    output = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt if negative_prompt else None,
    )
    return output.images[0]

with gr.Blocks() as demo:
    gr.Markdown("# F-Lite Diffusion WebUI")
    with gr.Row():
        with gr.Column():
            model_path = gr.Dropdown(
                choices=[
                    "checkpoints/F-Lite",
                    "checkpoints/F-Lite-Texture"
                ],
                value="checkpoints/F-Lite",
                label="Model"
            )
            prompt = gr.Textbox(label="Prompt", value="A photorealistic 3D render of a charming, mischievous young boy, approximately eight years old, possessing the endearingly unusual features of long, floppy donkey ears that droop playfully over his shoulders and a surprisingly small, pink pig nose that twitches slightly.  His eyes, a sparkling, intelligent hazel, are wide with a hint of playful mischief, framed by slightly unruly, sandy-brown hair that falls in tousled waves across his forehead.")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="distorted proportions, deformed, bad anatomy, extra limbs, missing limbs, disfigured face, poorly drawn hands, poorly rendered face, mutation")
            height = gr.Slider(256, 1536, value=1024, step=64, label="Height")
            width = gr.Slider(256, 1536, value=1024, step=64, label="Width")
            num_inference_steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
            guidance_scale = gr.Slider(1.0, 20.0, value=3.0, step=0.1, label="Guidance Scale")
            btn = gr.Button("Generate")
        with gr.Column():
            output_img = gr.Image(label="Generated Image")

    btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, guidance_scale, negative_prompt, model_path],
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")