import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModel
import os
import sys

# Add model path to system path
CHECKPOINT_PATH = "./checkpoints/NextStep-1.1"
sys.path.insert(0, CHECKPOINT_PATH)

from models.gen_pipeline import NextStepPipeline

# Initialize model and pipeline globally
print(f"Loading model from {CHECKPOINT_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT_PATH, 
    local_files_only=True, 
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    CHECKPOINT_PATH, 
    local_files_only=True, 
    trust_remote_code=True
)
pipeline = NextStepPipeline(
    tokenizer=tokenizer, 
    model=model,
    vae_name_or_path=os.path.join(CHECKPOINT_PATH, "vae")
).to(
    device="cuda", 
    dtype=torch.bfloat16
)
print("Model loaded successfully!")

# Create output directory if it doesn't exist
os.makedirs("./outputs", exist_ok=True)


def generate_image(
    prompt,
    positive_prompt,
    negative_prompt,
    img_size,
    num_images,
    cfg,
    cfg_img,
    num_sampling_steps,
    timesteps_shift,
    seed,
    use_norm,
    cfg_schedule,
):
    """Generate image based on the provided parameters"""
    try:
        # Generate image
        images = pipeline.generate_image(
            prompt,
            hw=(img_size, img_size),
            num_images_per_caption=num_images,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            cfg=cfg,
            cfg_img=cfg_img,
            cfg_schedule=cfg_schedule,
            use_norm=use_norm,
            num_sampling_steps=num_sampling_steps,
            timesteps_shift=timesteps_shift,
            seed=seed if seed != -1 else None,
        )
        
        # Save images
        output_images = []
        for idx, img in enumerate(images):
            output_path = f"./outputs/output_{seed}_{idx}.jpg"
            img.save(output_path)
            output_images.append(img)
        
        return output_images
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise gr.Error(f"生成图像时出错: {str(e)}")


# Create Gradio interface
with gr.Blocks(title="NextStep-1.1 Image Generator") as demo:
    gr.Markdown(
        """
        # NextStep-1.1 图像生成器
        基于 NextStep-1.1 模型的文本到图像生成界面
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            prompt = gr.Textbox(
                label="主提示词 (Main Prompt)",
                placeholder="输入你想要生成的图像描述...",
                lines=3,
                value="A REALISTIC PHOTOGRAPH OF A WALL WITH \"TOWARD AUTOREGRESSIVE IMAGE GENERATION WITH CONTINUOUS TOKENS AT SCALE\" PROMINENTLY DISPLAYED"
            )
            
            positive_prompt = gr.Textbox(
                label="正面提示词 (Positive Prompt)",
                placeholder="可选的正面提示词...",
                lines=2,
                value="high quality, detailed, 8k"
            )
            
            negative_prompt = gr.Textbox(
                label="负面提示词 (Negative Prompt)",
                lines=2,
                value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
            )
            
            with gr.Row():
                img_size = gr.Slider(
                    label="图像尺寸 (Image Size)",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=512
                )
                
                num_images = gr.Slider(
                    label="生成数量 (Number of Images)",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1
                )
            
            with gr.Row():
                cfg = gr.Slider(
                    label="CFG Scale",
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=7.5
                )
                
                cfg_img = gr.Slider(
                    label="CFG Image",
                    minimum=1.0,
                    maximum=5.0,
                    step=0.1,
                    value=1.0
                )
            
            with gr.Row():
                num_sampling_steps = gr.Slider(
                    label="采样步数 (Sampling Steps)",
                    minimum=10,
                    maximum=100,
                    step=1,
                    value=28
                )
                
                timesteps_shift = gr.Slider(
                    label="时间步偏移 (Timesteps Shift)",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0
                )
            
            with gr.Row():
                seed = gr.Number(
                    label="随机种子 (Seed, -1 for random)",
                    value=3407,
                    precision=0
                )
                
                cfg_schedule = gr.Dropdown(
                    label="CFG Schedule",
                    choices=["constant", "linear", "cosine"],
                    value="constant"
                )
            
            use_norm = gr.Checkbox(
                label="使用归一化 (Use Normalization)",
                value=False,
                info="对生成的token进行层归一化，可能提高生成稳定性，但通常保持关闭"
            )
            
            generate_btn = gr.Button("🎨 生成图像", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Output
            output_gallery = gr.Gallery(
                label="生成的图像",
                show_label=True,
                columns=1,
                rows=1,
                object_fit="contain",
                height=600,
                preview=True
            )
    
    # Examples section with descriptions
    gr.Markdown(
        """
        ---
        ## 📚 示例提示词 (Example Prompts)
        
        下方提供了多个精心设计的示例，用于测试 NextStep-1 模型的不同核心能力。点击任意示例即可自动填充！
        """
    )
    
    with gr.Row():
        example_category = gr.Dropdown(
            label="测试分类",
            choices=[
                "🔤 文字渲染能力",
                "📐 复杂空间关系", 
                "🔬 高保真细节",
                "🧠 逻辑推理",
                "🌄 经典场景"
            ],
            value="🔤 文字渲染能力",
            interactive=False,
            scale=1
        )
        example_translation = gr.Textbox(
            label="中文翻译",
            interactive=False,
            scale=2
        )
    
    gr.Examples(
        examples=[
            # 🔤 文字渲染能力测试
            [
                "🔤 文字渲染能力",
                "A futuristic neon sign hanging in a rainy cyber market that clearly spells out 'NextStep-1' in glowing blue letters.",
                "一个悬挂在雨中赛博市场的未来主义霓虹灯牌，用发光的蓝色字母清晰地拼写出 'NextStep-1'",
                3407,
            ],
            # 📐 复杂空间关系测试
            [
                "📐 复杂空间关系",
                "A wooden table with three distinct objects: a red apple on the far left, a vintage blue book in the center, and a glass of water on the right. Warm sunlight is casting shadows from the left side.",
                "一张木桌上有三个明显的物体：最左边是一个红苹果，中间是一本复古的蓝皮书，右边是一杯水。温暖的阳光从左侧投下阴影",
                123,
            ],
            # 🔬 高保真细节测试
            [
                "🔬 高保真细节",
                "Extreme macro photography of a dragonfly's eye, revealing the intricate hexagonal lattice structure, with iridescent colors reflecting a sunset.",
                "蜻蜓眼睛的极致微距摄影，展示出错综复杂的六边形晶格结构，并反射出日落的彩虹色光泽",
                789,
            ],
            # 🧠 逻辑推理测试
            [
                "🧠 逻辑推理",
                "An oil painting depicting Isaac Newton sitting under an apple tree, but the apple is a glowing holographic digital cube, symbolizing the transition from classical physics to the digital age.",
                "一幅描绘艾萨克·牛顿坐在苹果树下的油画，但苹果是一个发光的全息数字立方体，象征着从经典物理学到数字时代的转变",
                2024,
            ],
            # 🌄 经典场景测试
            [
                "🌄 经典场景",
                "A serene mountain landscape at sunset with vibrant orange and purple skies",
                "一幅宁静的山景，日落时分的天空呈现出鲜艳的橙色和紫色",
                1234,
            ],
        ],
        inputs=[
            example_category,
            prompt,
            example_translation,
            seed,
        ],
    )
    
    # Set up event handler
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            positive_prompt,
            negative_prompt,
            img_size,
            num_images,
            cfg,
            cfg_img,
            num_sampling_steps,
            timesteps_shift,
            seed,
            use_norm,
            cfg_schedule,
        ],
        outputs=output_gallery,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
