"""
SpotEdit Gradio Web 应用
基于 FLUX.1-Kontext 和 Qwen-Image-Edit 模型的图像编辑工具
"""

import os
import sys
import torch
import gradio as gr
from PIL import Image, ImageOps

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
# 添加子模块目录到路径，解决模块内部的非相对导入问题
sys.path.insert(0, os.path.join(ROOT_DIR, "FLUX_kontext"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Qwen_image_edit"))

from diffusers import FluxKontextPipeline, QwenImageEditPipeline
from diffusers.utils import load_image
from FLUX_kontext import generate as flux_generate, SpotEditConfig as FluxSpotEditConfig
from Qwen_image_edit import generate as qwen_generate, SpotEditConfig as QwenSpotEditConfig

# 模型路径配置
FLUX_MODEL_PATH = "./checkpoints/FLUX.1-Kontext-dev"
QWEN_MODEL_PATH = "./checkpoints/Qwen-Image-Edit"

# 示例图片路径
EXAMPLE_IMAGE_PATH = "./asset/dog.jpg"

# 全局变量存储管道
flux_pipe = None
qwen_pipe = None


def load_flux_model():
    """加载 FLUX 模型"""
    global flux_pipe
    if flux_pipe is None:
        print("正在加载 FLUX.1-Kontext 模型...")
        flux_pipe = FluxKontextPipeline.from_pretrained(
            FLUX_MODEL_PATH, 
            torch_dtype=torch.bfloat16
        ).to('cuda')
        print("FLUX.1-Kontext 模型加载完成！")
    return flux_pipe


def load_qwen_model():
    """加载 Qwen 模型"""
    global qwen_pipe
    if qwen_pipe is None:
        print("正在加载 Qwen-Image-Edit 模型...")
        qwen_pipe = QwenImageEditPipeline.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=torch.bfloat16
        ).to('cuda')
        print("Qwen-Image-Edit 模型加载完成！")
    return qwen_pipe


def flux_edit_image(
    image: Image.Image,
    prompt: str,
    threshold: float,
    num_inference_steps: int,
    guidance_scale: float,
):
    """使用 FLUX 模型编辑图像"""
    if image is None:
        return None, "请上传图片"
    
    if not prompt.strip():
        return None, "请输入编辑提示词"
    
    try:
        pipe = load_flux_model()
        
        # 调整图像大小，保留纵横比（中心裁剪并缩放），避免图片被拉伸变形
        image = ImageOps.fit(image, (1024, 1024), method=Image.BICUBIC)
        
        # 配置
        config = FluxSpotEditConfig(threshold=threshold)
        
        # 生成
        result = flux_generate(
            pipe,
            image=image,
            prompt=prompt,
            config=config,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        return result.images[0], "编辑完成！"
    except Exception as e:
        return None, f"错误: {str(e)}"


def qwen_edit_image(
    image: Image.Image,
    prompt: str,
    threshold: float,
    num_inference_steps: int,
):
    """使用 Qwen 模型编辑图像"""
    if image is None:
        return None, "请上传图片"
    
    if not prompt.strip():
        return None, "请输入编辑提示词"
    
    try:
        pipe = load_qwen_model()
        
        # 调整图像大小，保留纵横比（中心裁剪并缩放），避免图片被拉伸变形
        image = ImageOps.fit(image, (1024, 1024), method=Image.BICUBIC)
        
        # 配置
        config = QwenSpotEditConfig(threshold=threshold)
        
        # 生成
        result = qwen_generate(
            pipe,
            image=image,
            prompt=prompt,
            config=config,
            num_inference_steps=num_inference_steps,
        )
        
        return result.images[0], "编辑完成！"
    except Exception as e:
        return None, f"错误: {str(e)}"


# 示例数据
flux_examples = [
    [EXAMPLE_IMAGE_PATH, "add a scarf to the dog", 0.2, 50, 7.5],
]

qwen_examples = [
    [EXAMPLE_IMAGE_PATH, "add a scarf to the dog", 0.15, 50],
]


# 创建 Gradio 界面
def create_app():
    with gr.Blocks(
        title="SpotEdit - AI 图像编辑工具",
        theme=gr.themes.Soft(),
        css="""
        .header-banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .header-banner h1 {
            color: white;
            margin: 0;
            font-size: 2em;
        }
        .header-banner a {
            color: #ffd700;
            text-decoration: none;
            font-weight: bold;
        }
        .header-banner a:hover {
            text-decoration: underline;
        }
        """
    ) as app:
        # 顶部横幅 - YouTube 频道信息
        gr.HTML("""
        <div class="header-banner">
            <h1>🎨 SpotEdit - AI 智能图像编辑工具</h1>
            <p style="color: white; margin: 10px 0;">
                📺 欢迎访问我的 YouTube 频道: 
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">
                    AI 技术分享频道
                </a>
            </p>
            <p style="color: #e0e0e0; font-size: 0.9em;">
                基于 FLUX.1-Kontext 和 Qwen-Image-Edit 模型的精准图像编辑
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # FLUX 标签页
            with gr.TabItem("🌟 FLUX.1-Kontext 模型"):
                gr.Markdown("""
                ### 使用说明
                1. 上传一张图片
                2. 输入编辑提示词（描述你想要的修改）
                3. 调整参数（可选）
                4. 点击"开始编辑"按钮
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        flux_input_image = gr.Image(
                            label="输入图片",
                            type="pil",
                            height=400,
                        )
                        flux_prompt = gr.Textbox(
                            label="编辑提示词",
                            placeholder="例如: add a scarf to the dog",
                            lines=2,
                        )
                        
                        with gr.Accordion("高级参数", open=False):
                            flux_threshold = gr.Slider(
                                label="编辑阈值 (Threshold)",
                                minimum=0.05,
                                maximum=0.5,
                                value=0.2,
                                step=0.05,
                                info="控制编辑区域的大小，值越大编辑范围越小",
                            )
                            flux_steps = gr.Slider(
                                label="推理步数 (Steps)",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5,
                                info="步数越多质量越高，但速度越慢",
                            )
                            flux_guidance = gr.Slider(
                                label="引导系数 (Guidance Scale)",
                                minimum=1.0,
                                maximum=15.0,
                                value=7.5,
                                step=0.5,
                                info="控制提示词的影响强度",
                            )
                        
                        flux_submit_btn = gr.Button(
                            "🚀 开始编辑",
                            variant="primary",
                            size="lg",
                        )
                    
                    with gr.Column(scale=1):
                        flux_output_image = gr.Image(
                            label="编辑结果",
                            type="pil",
                            height=400,
                        )
                        flux_status = gr.Textbox(
                            label="状态",
                            interactive=False,
                        )
                
                # FLUX 示例
                gr.Examples(
                    examples=flux_examples,
                    inputs=[
                        flux_input_image,
                        flux_prompt,
                        flux_threshold,
                        flux_steps,
                        flux_guidance,
                    ],
                    outputs=[flux_output_image, flux_status],
                    fn=flux_edit_image,
                    cache_examples=False,
                    label="示例",
                )
                
                # FLUX 按钮点击事件
                flux_submit_btn.click(
                    fn=flux_edit_image,
                    inputs=[
                        flux_input_image,
                        flux_prompt,
                        flux_threshold,
                        flux_steps,
                        flux_guidance,
                    ],
                    outputs=[flux_output_image, flux_status],
                )
            
            # Qwen 标签页
            with gr.TabItem("🎯 Qwen-Image-Edit 模型"):
                gr.Markdown("""
                ### 使用说明
                1. 上传一张图片
                2. 输入编辑提示词（描述你想要的修改）
                3. 调整参数（可选）
                4. 点击"开始编辑"按钮
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        qwen_input_image = gr.Image(
                            label="输入图片",
                            type="pil",
                            height=400,
                        )
                        qwen_prompt = gr.Textbox(
                            label="编辑提示词",
                            placeholder="例如: add a scarf to the dog",
                            lines=2,
                        )
                        
                        with gr.Accordion("高级参数", open=False):
                            qwen_threshold = gr.Slider(
                                label="编辑阈值 (Threshold)",
                                minimum=0.05,
                                maximum=0.5,
                                value=0.15,
                                step=0.05,
                                info="控制编辑区域的大小，值越大编辑范围越小",
                            )
                            qwen_steps = gr.Slider(
                                label="推理步数 (Steps)",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5,
                                info="步数越多质量越高，但速度越慢",
                            )
                        
                        qwen_submit_btn = gr.Button(
                            "🚀 开始编辑",
                            variant="primary",
                            size="lg",
                        )
                    
                    with gr.Column(scale=1):
                        qwen_output_image = gr.Image(
                            label="编辑结果",
                            type="pil",
                            height=400,
                        )
                        qwen_status = gr.Textbox(
                            label="状态",
                            interactive=False,
                        )
                
                # Qwen 示例
                gr.Examples(
                    examples=qwen_examples,
                    inputs=[
                        qwen_input_image,
                        qwen_prompt,
                        qwen_threshold,
                        qwen_steps,
                    ],
                    outputs=[qwen_output_image, qwen_status],
                    fn=qwen_edit_image,
                    cache_examples=False,
                    label="示例",
                )
                
                # Qwen 按钮点击事件
                qwen_submit_btn.click(
                    fn=qwen_edit_image,
                    inputs=[
                        qwen_input_image,
                        qwen_prompt,
                        qwen_threshold,
                        qwen_steps,
                    ],
                    outputs=[qwen_output_image, qwen_status],
                )
        
        # 底部信息
        gr.Markdown("""
        ---
        ### 关于
        - **FLUX.1-Kontext**: Black Forest Labs 出品的高质量图像编辑模型
        - **Qwen-Image-Edit**: 阿里巴巴通义千问团队出品的图像编辑模型
        - **SpotEdit**: 精准的局部编辑技术，只修改需要改变的区域
        
        💡 **提示**: 编辑提示词使用英文效果更好
        """)
    
    return app


if __name__ == "__main__":
    print("=" * 50)
    print("SpotEdit - AI 智能图像编辑工具")
    print("=" * 50)
    print(f"FLUX 模型路径: {FLUX_MODEL_PATH}")
    print(f"Qwen 模型路径: {QWEN_MODEL_PATH}")
    print("=" * 50)
    
    # 预加载模型（可选，取消注释以在启动时加载）
    print("正在预加载模型...")
    load_flux_model()
    load_qwen_model()
    print("模型加载完成！")
    
    # 创建并启动应用
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
