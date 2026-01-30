import torch
import gradio as gr
from diffusers import ZImagePipeline
import random

# 模型路径
MODEL_PATH = "/workspace/zimage/checkpoints/Z-Image"

# 全局变量存储 pipeline
pipe = None

def load_model():
    """加载模型"""
    global pipe
    print("正在加载 Z-Image 模型...")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    print("模型加载完成！")
    return pipe

def generate_image(
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    randomize_seed,
):
    """生成图像"""
    global pipe
    
    if pipe is None:
        return None, "错误：模型未加载，请重启应用"
    
    # 处理随机种子
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    
    # 生成图像
    generator = torch.Generator("cuda").manual_seed(int(seed))
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=int(height),
        width=int(width),
        cfg_normalization=True,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    ).images[0]
    
    return image, f"生成成功！使用种子：{seed}"

# 示例提示词
EXAMPLES = [
    # 示例 1：地狱级双语排版
    [
        """A cinematic wide shot of a bustling ancient Chinese street mixed with high-tech elements.
To the left, a wooden shop features a hanging signboard with the text "云存储".[1] Inside, blue server racks glow.
[1] To the right, a shop features a sign reading "云计算".[1] Next to it, another signboard displays the text "云模型".[1] In front, a large brown ceramic wine vat has a red paper square with the bold calligraphy text "千问".
8k resolution, photorealistic, contrast between ancient architecture and cyberpunk neon lights, tyndall effect.""",
        "",  # negative_prompt
        1024,  # width
        1024,  # height
        4.0,  # guidance_scale
        50,  # num_inference_steps
        42,  # seed
        False,  # randomize_seed
    ],
    # 示例 2：微距材质与解剖学
    [
        """A close-up, dramatic portrait of a woman in a deep blue velvet dress, focused on her hands as she gently holds three glowing glass orbs.
The intricate detail of her fingers and knuckles is perfectly rendered.[2] Backlit, moody atmosphere, renaissance oil painting style, volumetric lighting, photorealistic skin texture, pores visible.""",
        "",
        1024,
        1024,
        4.0,
        50,
        42,
        False,
    ],
    # 示例 3：视觉错位与逻辑推理
    [
        """A surreal creative shot. A hand holding a smartphone horizontally.
On the screen, a cute girl wearing black-rimmed glasses is stepping out of the phone display. Her upper body is outside the screen in the real world 3D space, while her feet are still inside the screen digital world.
Strong forced perspective, shallow depth of field, clean grey background, studio lighting, hyper-realistic.""",
        "",
        720,
        1280,
        4.0,
        50,
        42,
        False,
    ],
    # 示例 4：透明材质与光影遮挡
    [
        """A black and white photograph with selective color. A blurred silhouette of a mysterious figure standing behind a frosted glass door. One hand is pressed sharply against the glass, creating a high-contrast, clear print detail. A distinct, bright yellow sticky note is attached to the frosted glass surface, standing out against the monochrome background. Cinematic lighting, noir atmosphere, 8k, highly detailed texture of the glass surface.""",
        "",
        1024,
        1024,
        4.0,
        50,
        42,
        False,
    ],
]

# 示例名称（用于显示）
EXAMPLE_NAMES = [
    "🏮 地狱级双语排版 - 论文图示同款（测试多路标+中文渲染）",
    "🖐️ 微距材质与解剖学 - 专治AI手（测试皮肤毛孔、丝绒材质、手部细节）",
    "📱 视觉错位与逻辑推理 - 3D破壁效果（测试空间逻辑理解）",
    "🪟 透明材质与光影遮挡 - 物理渲染极限（测试磨砂玻璃光影）",
]

def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="Z-Image 图像生成", theme=gr.themes.Soft()) as demo:
        # 顶部频道信息
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">📺 欢迎关注我的 YouTube 频道</h2>
            <h3 style="color: #f0f0f0; margin: 10px 0;">AI 技术分享频道</h3>
            <a href="https://www.youtube.com/@rongyi-ai" target="_blank" style="display: inline-block; padding: 10px 30px; background: #ff0000; color: white; text-decoration: none; border-radius: 25px; font-weight: bold; margin-top: 10px;">
                🔔 订阅频道
            </a>
        </div>
        """)
        
        # 标题
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🎨 Z-Image 图像生成演示</h1>
            <p style="color: #666;">基于通义万象 Z-Image 模型的文本到图像生成</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                prompt = gr.Textbox(
                    label="提示词",
                    placeholder="请输入图像描述...",
                    lines=5,
                )
                
                negative_prompt = gr.Textbox(
                    label="负面提示词（可选）",
                    placeholder="输入不希望出现的内容...",
                    lines=2,
                )
                
                with gr.Row():
                    width = gr.Slider(
                        label="宽度",
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                    )
                    height = gr.Slider(
                        label="高度",
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                    )
                
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="引导系数 (Guidance Scale)",
                        minimum=1.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.5,
                    )
                    num_inference_steps = gr.Slider(
                        label="推理步数",
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=1,
                    )
                
                with gr.Row():
                    seed = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0,
                    )
                    randomize_seed = gr.Checkbox(
                        label="随机种子",
                        value=False,
                    )
                
                generate_btn = gr.Button("🚀 生成图像", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # 输出区域
                output_image = gr.Image(
                    label="生成结果",
                    type="pil",
                )
                status_text = gr.Textbox(
                    label="状态",
                    interactive=False,
                )
        
        # 示例区域
        gr.HTML("""
        <div style="margin-top: 30px; margin-bottom: 10px;">
            <h3>📝 示例提示词</h3>
            <p style="color: #666;">点击下方示例可快速加载预设的提示词和参数</p>
        </div>
        """)
        
        # 示例说明
        with gr.Accordion("💡 示例说明", open=False):
            gr.Markdown("""
### 1. 🏮 地狱级双语排版 - 论文图示同款
这是 Z-Image 最骄傲的"多路标+中文渲染"能力。普通模型画一个招牌还行，画一条街的招牌通常会乱码。
- **测试点**：左边"云存储"，右边"云计算"、"千问"。看字写得对不对，位置乱不乱。

### 2. 🖐️ 微距材质与解剖学 - 专治"AI手"
论文中特意强调了 Z-Image 在解剖学（Anatomy）和高频细节（Texture）上的优化。
- **测试点**：手指是否畸形，丝绒衣服的毛绒感是否真实，光影是否像油画一样高级。

### 3. 📱 视觉错位与逻辑推理 - 3D破壁效果
Z-Image 引入了"Prompt Enhancer"（提示词增强）和推理能力。这种"画中画"或者"打破第四面墙"的构图，非常考验模型对空间逻辑的理解。
- **测试点**：手机屏幕内外的透视关系，人物是否真的像"钻出来"一样。

### 4. 🪟 透明材质与光影遮挡 - 物理渲染极限
这是论文中提到的 S3-DiT 架构对物理光影模拟的优势。
- **测试点**：磨砂玻璃后的模糊剪影 vs 贴在玻璃上的清晰手印/物体。
            """)
        
        gr.Examples(
            examples=EXAMPLES,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                randomize_seed,
            ],
            outputs=[output_image, status_text],
            fn=generate_image,
            cache_examples=False,
            examples_per_page=4,
            label="点击示例加载预设",
        )
        
        # 绑定生成按钮
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                randomize_seed,
            ],
            outputs=[output_image, status_text],
        )
        
        # 页脚
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #eee;">
            <p style="color: #888;">
                推荐参数：分辨率 512×512 到 2048×2048 | 引导系数 3.0-5.0 | 推理步数 28-50
            </p>
            <p style="color: #888;">
                基于 <a href="https://github.com/Tongyi-MAI/Z-Image" target="_blank">Tongyi-MAI/Z-Image</a> 模型
            </p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    # 启动时加载模型
    load_model()
    
    # 创建并启动 UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
