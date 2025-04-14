import torch
import gradio as gr
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    FluxTransformer2DModel,
    FluxImg2ImgPipeline,
    FluxPipeline,
    StableDiffusionPipeline
)
from transformers import BitsAndBytesConfig, T5EncoderModel, T5TokenizerFast

# 全局变量存储pipelines
t2i_pipeline = None
i2i_pipeline = None

def load_t2i_model():
    global t2i_pipeline
    if t2i_pipeline is not None:
        return t2i_pipeline
        
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    text_encoder_8bit = T5EncoderModel.from_pretrained(
        "checkpoints/FLUX.1-dev",
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = FluxTransformer2DModel.from_pretrained(
        "checkpoints/FLUX.1-dev",
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )

    t2i_pipeline = FluxPipeline.from_pretrained(
        "checkpoints/FLUX.1-dev",
        text_encoder_2=text_encoder_8bit,
        transformer=transformer_8bit,
        torch_dtype=torch.float16,
        device_map="balanced",
    )
    
    t2i_pipeline.load_lora_weights("checkpoints/flux-chatgpt-ghibli-lora", weight_name="flux-chatgpt-ghibli-lora.safetensors")
    return t2i_pipeline

def load_i2i_model():
    global i2i_pipeline
    if i2i_pipeline is not None:
        return i2i_pipeline
    
    i2i_pipeline = StableDiffusionPipeline.from_pretrained(
        "checkpoints/Ghibli-Diffusion",
        torch_dtype=torch.float16
    ).to("cuda")
    return i2i_pipeline

def t2i_generate(prompt):
    global t2i_pipeline
    t2i_pipeline = load_t2i_model()
    
    try:
        output = t2i_pipeline(
            prompt,
            guidance_scale=3.5,
            height=768,
            width=1360,
            num_inference_steps=50
        ).images[0]
        
        # 只清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        return output
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

def i2i_generate(prompt, cfg_scale=7.5, steps=25, seed=None, width=512, height=512, negative_prompt=""):
    global i2i_pipeline
    i2i_pipeline = load_i2i_model()
    
    if seed is not None and seed != "":
        generator = torch.Generator("cuda").manual_seed(int(seed))
    else:
        generator = None
    
    return i2i_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg_scale),
        height=int(height),
        width=int(width),
        generator=generator
    ).images[0]

# 添加示例提示词列表
examples = [
    "Ghibli style futuristic stormtrooper with glossy white armor and a sleek helmet, standing heroically on a lush alien planet, vibrant flowers blooming around, soft sunlight illuminating the scene, a gentle breeze rustling the leaves. The armor reflects the pink and purple hues of the alien sunset, creating an ethereal glow around the figure. [trigger]",
    "Ghibli style young mechanic girl in a floating workshop, surrounded by hovering tools and glowing mechanical parts, her blue overalls covered in oil stains, tinkering with a semi-transparent robot companion. Magical sparks fly as she works, while floating islands with waterfalls drift past her open workshop window. [trigger]",
    "Ghibli style ancient forest guardian robot, covered in moss and flowering vines, sitting peacefully in a crystal-clear lake. Its gentle eyes glow with soft blue light, while bioluminescent dragonflies dance around its weathered metal frame. Ancient tech symbols on its surface pulse with a gentle rhythm. [trigger]",
    "Ghibli style sky whale transport ship, its metallic skin adorned with traditional Japanese patterns, gliding through cotton candy clouds at sunrise. Small floating gardens hang from its sides, where workers in futuristic kimonos tend to glowing plants. Rainbow auroras shimmer in the background. [trigger]",
    "Ghibli style cyber-shrine maiden with flowing holographic robes, performing a ritual dance among floating lanterns and digital cherry blossoms. Her traditional headdress emits soft light patterns, while spirit-like AI constructs swirl around her in elegant patterns. The scene is set in a modern shrine with both ancient wood and sleek chrome elements. [trigger]",
    "Ghibli style robot farmer tending to floating rice paddies in the sky, wearing a traditional straw hat with advanced sensors. Its gentle movements create ripples in the water as it plants glowing rice seedlings. Flying fish leap between the terraced fields, leaving trails of sparkles in their wake, while future Tokyo's spires gleam in the distance. [trigger]"
]

# 更新 SD 示例提示词列表，包含完整参数
sd_examples = [
    {
        "prompt": "ghibli style ice field white mountains ((northern lights)) starry sky low horizon",
        "negative_prompt": "soft blurry",
        "steps": 25,
        "cfg_scale": 7.5,
        "seed": "",
        "width": 512,
        "height": 512
    },
    {
        "prompt": "ghibli style (storm trooper)",
        "negative_prompt": "(bad anatomy)",
        "steps": 20,
        "cfg_scale": 7.0,
        "seed": 3450349066,
        "width": 512,
        "height": 704
    },
    {
        "prompt": "ghibli style VW beetle",
        "negative_prompt": "soft blurry",
        "steps": 30,
        "cfg_scale": 7.0,
        "seed": 1529856912,
        "width": 704,
        "height": 512
    }
]

with gr.Blocks() as demo:
    gr.Markdown("# Ghibli Style Image Generator")
    
    with gr.Tabs() as tabs:
        with gr.Tab("Flux LoRA"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(
                        label="输入提示词",
                        placeholder="请输入图片描述...",
                        lines=3
                    )
                    t2i_btn = gr.Button("生成图片")
                
                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="生成的图片")
            
            with gr.Row():
                examples_table = gr.Dataset(
                    components=[t2i_prompt],
                    samples=[[example] for example in examples],
                    headers=["提示词示例"]
                )
            
            examples_table.click(
                fn=lambda x: x[0],
                inputs=[examples_table],
                outputs=t2i_prompt
            )
            
            t2i_btn.click(
                fn=t2i_generate,
                inputs=t2i_prompt,
                outputs=t2i_output,
                api_name=False
            )
            
        with gr.Tab("StableDiffusion"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_prompt = gr.Textbox(label="提示词", lines=2)
                    i2i_negative = gr.Textbox(label="反向提示词", lines=2)
                    i2i_cfg = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="CFG Scale")
                    i2i_steps = gr.Slider(minimum=1, maximum=50, value=25, step=1, label="Steps")
                    with gr.Row():
                        i2i_width = gr.Number(label="Width", value=512, precision=0)
                        i2i_height = gr.Number(label="Height", value=512, precision=0)
                    i2i_seed = gr.Textbox(label="Seed (空为随机)", value="")
                    i2i_btn = gr.Button("生成图片")
                
                with gr.Column(scale=1):
                    i2i_output = gr.Image(label="生成的图片")
            
            with gr.Row():
                sd_examples_table = gr.Dataset(
                    components=[i2i_prompt, i2i_negative, i2i_cfg, i2i_steps, i2i_width, i2i_height, i2i_seed],
                    samples=[[ex["prompt"], ex["negative_prompt"], ex["cfg_scale"], ex["steps"], 
                             ex["width"], ex["height"], ex["seed"]] for ex in sd_examples],
                    headers=["提示词示例"]
                )
            
            sd_examples_table.click(
                fn=lambda x: x,
                inputs=[sd_examples_table],
                outputs=[i2i_prompt, i2i_negative, i2i_cfg, i2i_steps, i2i_width, i2i_height, i2i_seed]
            )
            
            i2i_btn.click(
                fn=i2i_generate,
                inputs=[
                    i2i_prompt, i2i_cfg, i2i_steps, i2i_seed,
                    i2i_width, i2i_height, i2i_negative
                ],
                outputs=i2i_output,
                api_name=False
            )

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', share=False)
