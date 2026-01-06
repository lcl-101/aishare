import torch
import math
import time
import gradio as gr
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

# 配置
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = "./checkpoints/Qwen-Image-2512"
lora_path = "./checkpoints/Qwen-Image-2512-Turbo-LoRA"
lora_weight_name = "Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V2.0-bf16.safetensors"

# Scheduler配置
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

print("正在加载模型...")
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipe = DiffusionPipeline.from_pretrained(
    base_model, scheduler=scheduler, torch_dtype=dtype
).to(device)
print("基础模型加载完成！")

print("正在加载LoRA权重...")
pipe.load_lora_weights(
    lora_path,
    weight_name=lora_weight_name,
    adapter_name="lightning"
)
print("LoRA权重加载完成！模型初始化完毕！")


def generate_images(prompt, guidance_scale=1.0, seed=-1):
    """同时使用两个模型生成图像"""
    if seed == -1:
        seed = torch.randint(0, 999999, (1,)).item()
    
    generator_base = torch.Generator(device=device).manual_seed(seed)
    generator_turbo = torch.Generator(device=device).manual_seed(seed)
    
    # 禁用LoRA，使用基础模型生成（50步）
    pipe.disable_lora()
    start_time = time.time()
    result_base = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        generator=generator_base,
    )
    time_base = time.time() - start_time
    
    # 启用LoRA，使用Turbo模型生成（4步）
    pipe.enable_lora()
    start_time = time.time()
    result_turbo = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=guidance_scale,
        generator=generator_turbo,
    )
    time_turbo = time.time() - start_time
    
    info_base = f"⏱️ 推理时间: {time_base:.2f}秒 | 步数: 50"
    info_turbo = f"⏱️ 推理时间: {time_turbo:.2f}秒 | 步数: 4 | 🚀 加速比: {time_base/time_turbo:.1f}x"
    
    return result_base.images[0], info_base, result_turbo.images[0], info_turbo


# 示例提示词
examples = [
    ["ultra-realistic 3D render of four mechanical keyboard keycaps in a tight 2x2 grid, all keys touching. View from an isometric angle. One key is transparent with the word \"Qwen\" printed in white key. The other three colors are: black, purple, and white. The black key says the white \"Image\" . The other two say \"25\" and \"12\". Realistic plastic texture, rounded sculpted keycaps, soft shadows, clean light-gray background."],
    ["a young girl with flowing long hair, wearing a white halter dress and smiling sweetly. The background features a blue seaside where seagulls fly freely."],
    ["A dreamy and ethereal hand-drawn flat illustration in a Post-Impressionist style, featuring impressionistic brushwork and abstract, minimalist lines. A close-up view shows a little boy in plush pajamas balancing on a ladder made of clouds in the night sky. He is hanging freshly washed, wet stars that are dripping liquid light, one by one, onto a long clothesline strung between the tips of a crescent moon. Beside him, a glowing little rabbit is helping by handing him clothespins. The scene is filled with bright, vibrant colors, bokeh brushstrokes, washes of pale golden mist, soft textures, and gentle soft lighting with a soft focus effect."],
    ["Bookstore window display. A sign displays \"New Arrivals This Week\". Below, a shelf tag with the text \"Best-Selling Novels Here\". To the side, a colorful poster advertises \"Author Meet And Greet on Saturday\" with a central portrait of the author. There are four books on the bookshelf, namely \"The light between worlds\" \"When stars are scattered\" \"The slient patient\" \"The night circus\""],
    ["A four-panel sci-fi comedy comic strip, vertical layout. The style mixes futuristic cyberpunk elements with a mundane kitchen setting. Bright neon accents.\nPanel 1 (Top): A sleek, advanced humanoid robot with glowing blue eyes stands in a normal kitchen, wearing a \"KISS THE COOK\" apron. It holds a spatula dramatically.\nText bubble (Robot, robotic font): \"任务已接受：正在执行'制作煎蛋'程序。成功率计算中：99.9%。\" (Task Accepted: Executing 'Make Omelet' protocol. Calculating success rate: 99.9%.)\nPanel 2: The robot is staring intensely at a carton of eggs. Its eyes are projecting complex holographic scanning grids and analytical data over a single egg.\nText bubble (Robot thinking): \"分析蛋壳结构……探测微小裂缝……优化敲击力度矢量。\" (Analyzing shell structure... detecting micro-fractures... optimizing impact force vectors.)\nPanel 3: CHAOS. The robot uses way too much force or advanced weaponry. It is firing a miniature laser beam from its finger at the egg, which has exploded into a cloud of shell and yolk. The kitchen is covered in mess.\nText bubble (Sound effect, huge): \"轰！！\" (BOOM!!)\nText bubble (Robot): \"哎呀。\" (Oops.)\nPanel 4 (Bottom): The robot stands covered in egg yolk, looking dejected. On the plate is a tiny, charred, unrecognizable black crisp.\nText bubble (Robot): \"任务失败。重新计算成功率：0.01%。我需要下载'常识'补丁。\" (Task Failed. Recalculating success rate: 0.01%. I need to download the 'Common Sense' patch.)"],
]

# 创建Gradio界面
with gr.Blocks(title="Qwen-Image-2512 Turbo 对比生成") as demo:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🎨 Qwen-Image-2512 vs Turbo 对比生成器</h1>
            <p style="font-size: 16px;">
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="text-decoration: none;">
                    <strong>📺 AI 技术分享频道</strong>
                </a>
            </p>
            <p style="color: #666;">同时对比基础模型（50步）和Turbo加速模型（4步）的生成效果</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="提示词",
                placeholder="请输入您想要生成的图像描述...",
                lines=5
            )
            
            with gr.Accordion("高级设置", open=False):
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    label="引导强度"
                )
                seed = gr.Slider(
                    minimum=-1,
                    maximum=999999,
                    value=-1,
                    step=1,
                    label="随机种子（-1表示随机）"
                )
            
            generate_btn = gr.Button("🎨 生成对比图像", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            output_image_base = gr.Image(label="📊 基础模型 (Qwen-Image-2512)", type="pil")
            info_base = gr.Textbox(label="性能信息", interactive=False)
        
        with gr.Column():
            output_image_turbo = gr.Image(label="🚀 Turbo模型 (带LoRA加速)", type="pil")
            info_turbo = gr.Textbox(label="性能信息", interactive=False)
    
    # 示例
    gr.Examples(
        examples=examples,
        inputs=prompt_input,
        label="示例提示词"
    )
    
    # 绑定生成函数
    generate_btn.click(
        fn=generate_images,
        inputs=[prompt_input, guidance_scale, seed],
        outputs=[output_image_base, info_base, output_image_turbo, info_turbo]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
