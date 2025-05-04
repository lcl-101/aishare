import gradio as gr
from src.pipeline import load_drape, infer_drape
from transparent_background import Remover
from PIL import Image

# 初始化 pipeline 和 remover
pipeline = load_drape()
remover = Remover()

def generate(prompt, image_ref, seed):
    if image_ref is None:
        return None
    image_ref = image_ref.convert("RGB")
    image = infer_drape(
        pipe=pipeline,
        prompt=prompt,
        image_ref=image_ref,
        remover=remover,
        seed=seed
    )
    return image

with gr.Blocks() as demo:
    gr.Markdown("# Uwear Drape1 WebUI Demo")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="A woman posing for a photoshoot, wearing a 'uwear.ai' white tshirt")
            image_ref = gr.Image(label="Reference Image", type="pil")
            seed = gr.Number(label="Seed", value=42, precision=0)
            btn = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(label="Generated Image")

    btn.click(
        generate,
        inputs=[prompt, image_ref, seed],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")