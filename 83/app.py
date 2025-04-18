import torch
from PIL import Image
from diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig as TransformersBitsAndBytesConfig
import gradio as gr

# 路径配置
LLAMA_PATH = "checkpoints/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
HIDREAM_PATH = "checkpoints/HiDream-I1-Full-nf4"

device = "cuda" if torch.cuda.is_available() else "cpu"

# HiDream 量化配置
hidream_quant_config = BitsAndBytesConfig(load_in_4bit=True)

# 加载 Llama Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, use_fast=False)
text_encoder = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device,
    low_cpu_mem_usage=True,
    output_hidden_states=True,
    output_attentions=True,
)

# 加载 HiDream Transformer
transformer = HiDreamImageTransformer2DModel.from_pretrained(
    HIDREAM_PATH,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    quantization_config=hidream_quant_config,
    device_map=device,
)

# 构建 pipeline
pipe = HiDreamImagePipeline.from_pretrained(
    HIDREAM_PATH,
    transformer=transformer,
    tokenizer_4=tokenizer,
    text_encoder_4=text_encoder,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)

def infer(
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 30,
    seed: int = 42
):
    generator = torch.Generator(device).manual_seed(seed)
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        image = result.images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("# HiDream WebUI")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="spider man")
            height = gr.Number(label="Height", value=1024, precision=0)
            width = gr.Number(label="Width", value=1024, precision=0)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=5.0, step=0.1)
            num_inference_steps = gr.Slider(label="Num Inference Steps", minimum=1, maximum=100, value=30, step=1)
            seed = gr.Number(label="Seed", value=42, precision=0)
            btn = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(label="Generated Image")

    btn.click(
        fn=infer,
        inputs=[prompt, height, width, guidance_scale, num_inference_steps, seed],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')

# 清理显存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
import gc
gc.collect()