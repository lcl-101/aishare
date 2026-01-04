import torch
import time
import gradio as gr
from diffusers import Flux2Pipeline

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

# 模型路径
MODEL_PATH = "checkpoints/FLUX.2-dev"
LORA_PATH = "checkpoints/FLUX.2-dev-Turbo/flux.2-turbo-lora.safetensors"

# 全局变量存储管道
pipe = None
lora_loaded = False

def load_models():
    """启动时加载模型"""
    global pipe, lora_loaded
    
    print("正在加载 FLUX.2-dev 基础模型...")
    pipe = Flux2Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("正在预加载 Turbo LoRA 权重...")
    pipe.load_lora_weights(LORA_PATH)
    lora_loaded = True
    
    print("模型加载完成！")

def generate_comparison(
    prompt: str,
    height: int,
    width: int,
    guidance_scale_standard: float,
    guidance_scale_turbo: float,
    seed: int,
    progress=gr.Progress()
):
    """生成标准50步和Turbo 8步的对比图像"""
    global pipe, lora_loaded
    
    if pipe is None:
        return None, None, "错误：模型未加载，请重启应用"
    
    # 设置随机种子
    generator_standard = torch.Generator("cuda").manual_seed(seed)
    generator_turbo = torch.Generator("cuda").manual_seed(seed)
    
    # 确保 LoRA 未融合，使用基础模型
    if lora_loaded:
        progress(0.05, desc="正在准备标准模式...")
        pipe.unfuse_lora()
    
    progress(0.1, desc="正在使用标准模式生成（50步）...")
    # 标准50步推理 - 记录耗时
    start_time_standard = time.time()
    image_standard = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale_standard,
        height=height,
        width=width,
        num_inference_steps=50,
        generator=generator_standard,
    ).images[0]
    time_standard = time.time() - start_time_standard
    
    # 融合 Turbo LoRA
    progress(0.55, desc="正在准备 Turbo 模式...")
    pipe.fuse_lora()
    
    progress(0.6, desc="正在使用Turbo模式生成（8步）...")
    # Turbo 8步推理 - 记录耗时
    start_time_turbo = time.time()
    image_turbo = pipe(
        prompt=prompt,
        sigmas=TURBO_SIGMAS,
        guidance_scale=guidance_scale_turbo,
        height=height,
        width=width,
        num_inference_steps=8,
        generator=generator_turbo,
    ).images[0]
    time_turbo = time.time() - start_time_turbo
    
    # 取消融合 LoRA，恢复基础模型状态
    pipe.unfuse_lora()
    
    progress(1.0, desc="生成完成！")
    
    # 计算加速比
    speedup = time_standard / time_turbo if time_turbo > 0 else 0
    
    info_text = f"""
    ✅ 生成完成！
    
    📝 提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}
    📐 分辨率: {width} x {height}
    🎲 随机种子: {seed}
    
    ⏱️ 标准模式: 50步 | 引导系数: {guidance_scale_standard} | 耗时: {time_standard:.2f} 秒
    ⚡ Turbo模式: 8步 | 引导系数: {guidance_scale_turbo} | 耗时: {time_turbo:.2f} 秒
    
    🚀 Turbo 加速比: {speedup:.2f}x (快了 {time_standard - time_turbo:.2f} 秒)
    """
    
    return image_standard, image_turbo, info_text

def generate_standard_only(
    prompt: str,
    height: int,
    width: int,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    progress=gr.Progress()
):
    """仅使用标准模式生成"""
    global pipe, lora_loaded
    
    if pipe is None:
        return None, "错误：模型未加载，请重启应用"
    
    # 确保 LoRA 未融合，使用基础模型
    if lora_loaded:
        pipe.unfuse_lora()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    progress(0.2, desc=f"正在使用标准模式生成（{num_steps}步）...")
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        generator=generator,
    ).images[0]
    
    progress(1.0, desc="生成完成！")
    
    info_text = f"✅ 标准模式生成完成 | 步数: {num_steps} | 种子: {seed}"
    return image, info_text

def generate_turbo_only(
    prompt: str,
    height: int,
    width: int,
    guidance_scale: float,
    seed: int,
    progress=gr.Progress()
):
    """仅使用Turbo模式生成"""
    global pipe, lora_loaded
    
    if pipe is None:
        return None, "错误：模型未加载，请重启应用"
    
    # 融合 Turbo LoRA
    if lora_loaded:
        progress(0.1, desc="正在准备 Turbo 模式...")
        pipe.fuse_lora()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    progress(0.2, desc="正在使用Turbo模式生成（8步）...")
    image = pipe(
        prompt=prompt,
        sigmas=TURBO_SIGMAS,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=8,
        generator=generator,
    ).images[0]
    
    progress(1.0, desc="生成完成！")
    
    # 取消融合 LoRA
    pipe.unfuse_lora()
    
    info_text = f"⚡ Turbo模式生成完成 | 步数: 8 | 种子: {seed}"
    return image, info_text

# 示例提示词
EXAMPLE_PROMPTS = [
    ["Industrial product shot of a chrome turbocharger with glowing hot exhaust manifold, engraved text 'FLUX.2 [dev] Turbo by fal' on the compressor housing and 'fal' on the turbine wheel, gradient heat glow from orange to electric blue , studio lighting with dramatic shadows, shallow depth of field, engineering blueprint pattern in background."],
    ["A majestic dragon soaring through a stormy sky, lightning crackling around its wings, scales shimmering with iridescent colors, epic fantasy art style, dramatic lighting, 8k ultra detailed"],
    ["Portrait of a cyberpunk samurai, neon lights reflecting off chrome armor, rain-soaked streets of Neo Tokyo in background, blade glowing with energy, cinematic composition, moody atmosphere"],
    ["A cozy coffee shop interior, warm golden hour sunlight streaming through large windows, steam rising from freshly brewed coffee, vintage wooden furniture, plants hanging from ceiling, photorealistic"],
    ["Underwater scene of a ancient sunken temple, bioluminescent jellyfish floating around marble columns, god rays piercing through crystal clear water, mystical and serene atmosphere"],
    ["A futuristic space station orbiting a gas giant planet, massive rings visible in background, astronauts performing spacewalk, Earth visible in distance, hard science fiction style, NASA-quality rendering"],
    ["Beautiful Japanese garden in autumn, red maple leaves falling gently, traditional wooden bridge over koi pond, misty morning atmosphere, zen aesthetic, highly detailed"],
    ["Steampunk airship floating above Victorian London, brass gears and copper pipes, steam billowing from engines, passengers on deck with period clothing, sunset colors, adventure feeling"],
]

# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="FLUX.2-dev Turbo 图像生成", theme=gr.themes.Soft()) as demo:
        # 顶部频道信息
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2em;">🎨 FLUX.2-dev Turbo 图像生成器</h1>
            <p style="color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.1em;">
                📺 欢迎访问我的 YouTube 频道: 
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="color: #ffeb3b; text-decoration: none; font-weight: bold;">
                    🎬 AI 技术分享频道
                </a>
            </p>
            <p style="color: #e0e0e0; margin: 5px 0 0 0; font-size: 0.9em;">
                对比体验官方 FLUX.2-dev 50步 与 Turbo LoRA 8步的生成效果
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # 对比模式标签页
            with gr.TabItem("🔄 对比模式", id="comparison"):
                gr.Markdown("### 同时生成标准50步和Turbo 8步的图像，方便对比效果")
                
                prompt_compare = gr.Textbox(
                    label="提示词",
                    placeholder="请输入图像描述...",
                    lines=3
                )
                
                with gr.Row():
                    width_compare = gr.Slider(
                        label="宽度",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024
                    )
                    height_compare = gr.Slider(
                        label="高度",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024
                    )
                    guidance_standard = gr.Slider(
                        label="标准模式引导系数",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=3.5
                    )
                    guidance_turbo = gr.Slider(
                        label="Turbo模式引导系数",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=2.5
                    )
                
                with gr.Row():
                    seed_compare = gr.Number(
                        label="随机种子（相同种子可复现结果）",
                        value=42,
                        precision=0
                    )
                    btn_compare = gr.Button("🚀 开始对比生成", variant="primary", size="lg", scale=2)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📷 标准模式 (50步)")
                        output_standard = gr.Image(label="标准50步结果", type="pil")
                    with gr.Column():
                        gr.Markdown("#### ⚡ Turbo模式 (8步)")
                        output_turbo = gr.Image(label="Turbo 8步结果", type="pil")
                
                info_compare = gr.Textbox(
                    label="生成信息",
                    interactive=False,
                    lines=6
                )
                
                # 示例
                gr.Markdown("### 📝 示例提示词（点击选择）")
                gr.Examples(
                    examples=EXAMPLE_PROMPTS,
                    inputs=[prompt_compare],
                    label=""
                )
                
                btn_compare.click(
                    fn=generate_comparison,
                    inputs=[prompt_compare, height_compare, width_compare, guidance_standard, guidance_turbo, seed_compare],
                    outputs=[output_standard, output_turbo, info_compare]
                )
            
            # 标准模式标签页
            with gr.TabItem("📷 标准模式", id="standard"):
                gr.Markdown("### 使用原版 FLUX.2-dev 模型生成图像（可自定义步数）")
                
                prompt_standard = gr.Textbox(
                    label="提示词",
                    placeholder="请输入图像描述...",
                    lines=3
                )
                
                with gr.Row():
                    width_standard = gr.Slider(
                        label="宽度",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024
                    )
                    height_standard = gr.Slider(
                        label="高度",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024
                    )
                    guidance_standard_only = gr.Slider(
                        label="引导系数",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=3.5
                    )
                    steps_standard = gr.Slider(
                        label="推理步数",
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=50
                    )
                
                with gr.Row():
                    seed_standard = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0
                    )
                    btn_standard = gr.Button("🎨 生成图像", variant="primary", size="lg", scale=2)
                
                output_standard_only = gr.Image(label="生成结果", type="pil")
                
                info_standard = gr.Textbox(
                    label="生成信息",
                    interactive=False,
                    lines=2
                )
                
                # 示例
                gr.Markdown("### 📝 示例提示词（点击选择）")
                gr.Examples(
                    examples=EXAMPLE_PROMPTS,
                    inputs=[prompt_standard],
                    label=""
                )
                
                btn_standard.click(
                    fn=generate_standard_only,
                    inputs=[prompt_standard, height_standard, width_standard, guidance_standard_only, steps_standard, seed_standard],
                    outputs=[output_standard_only, info_standard]
                )
            
            # Turbo模式标签页
            with gr.TabItem("⚡ Turbo模式", id="turbo"):
                gr.Markdown("### 使用 FLUX.2-dev-Turbo LoRA 快速生成图像（固定8步）")
                
                prompt_turbo = gr.Textbox(
                    label="提示词",
                    placeholder="请输入图像描述...",
                    lines=3
                )
                
                with gr.Row():
                    width_turbo = gr.Slider(
                        label="宽度",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024
                    )
                    height_turbo = gr.Slider(
                        label="高度",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024
                    )
                    guidance_turbo_only = gr.Slider(
                        label="引导系数",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=2.5
                    )
                
                with gr.Row():
                    seed_turbo = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0
                    )
                    btn_turbo = gr.Button("⚡ 快速生成", variant="primary", size="lg", scale=2)
                
                output_turbo_only = gr.Image(label="生成结果", type="pil")
                
                info_turbo = gr.Textbox(
                    label="生成信息",
                    interactive=False,
                    lines=2
                )
                
                # 示例
                gr.Markdown("### 📝 示例提示词（点击选择）")
                gr.Examples(
                    examples=EXAMPLE_PROMPTS,
                    inputs=[prompt_turbo],
                    label=""
                )
                
                btn_turbo.click(
                    fn=generate_turbo_only,
                    inputs=[prompt_turbo, height_turbo, width_turbo, guidance_turbo_only, seed_turbo],
                    outputs=[output_turbo_only, info_turbo]
                )
        
        # 底部说明
        gr.HTML("""
        <div style="text-align: center; padding: 15px; margin-top: 20px; background: #f5f5f5; border-radius: 10px;">
            <p style="margin: 0; color: #666;">
                💡 <strong>提示：</strong> Turbo模式使用特殊的sigma调度，仅需8步即可生成高质量图像，速度提升约6倍！
            </p>
            <p style="margin: 5px 0 0 0; color: #888; font-size: 0.9em;">
                模型: FLUX.2-dev + FLUX.2-dev-Turbo LoRA | 基于 Diffusers 库
            </p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 正在启动 FLUX.2-dev Turbo 图像生成器...")
    print("=" * 50)
    
    # 加载模型
    load_models()
    
    # 创建并启动界面
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
