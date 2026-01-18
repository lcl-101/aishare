import torch
import gradio as gr
from diffusers import Flux2KleinPipeline
import random
import time

# 设备和数据类型配置
device = "cuda"
dtype = torch.bfloat16

# 加载本地模型
print("正在加载模型...")
pipe = Flux2KleinPipeline.from_pretrained(
    "checkpoints/FLUX.2-klein-9B", 
    torch_dtype=dtype
)
pipe.enable_model_cpu_offload()
print("模型加载完成！")

# 文生图示例提示词
txt2img_example_prompts = [
    "A cat holding a sign that says hello world",
    "A futuristic cityscape at sunset with flying cars",
    "A magical forest with glowing mushrooms and fireflies",
    "A steampunk robot reading a book in a cozy library",
    "A majestic dragon flying over snow-capped mountains",
    "An astronaut playing guitar on the moon",
    "A beautiful Japanese garden with cherry blossoms and a koi pond",
    "A cyberpunk street market at night with neon lights",
    "A cute corgi wearing a tiny crown sitting on a throne",
    "A watercolor painting of a Venice canal at sunrise",
]

# 根据原始提示词生成相关的编辑提示词
def get_related_edit_prompts(original_prompt):
    """根据原始提示词生成相关的编辑提示词"""
    base_edits = [
        f"{original_prompt}, in the style of oil painting",
        f"{original_prompt}, in anime style",
        f"{original_prompt}, in watercolor style",
        f"{original_prompt}, at night with dramatic lighting",
        f"{original_prompt}, in cyberpunk style with neon colors",
        f"{original_prompt}, in vintage photograph style",
        f"{original_prompt}, in minimalist art style",
        f"{original_prompt}, with a dreamy, ethereal atmosphere",
    ]
    return base_edits

# 当前会话状态
current_state = {
    "original_prompt": "",
    "generated_image": None
}

def generate_image(prompt, height, width, guidance_scale, num_steps, seed):
    """文生图函数"""
    if not prompt.strip():
        return None, "请输入提示词！", gr.update()
    
    try:
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        start_time = time.time()
        
        image = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps),
            generator=generator
        ).images[0]
        
        elapsed_time = time.time() - start_time
        
        # 保存状态
        current_state["original_prompt"] = prompt
        current_state["generated_image"] = image
        
        # 生成相关编辑提示词
        related_prompts = get_related_edit_prompts(prompt)
        
        status_msg = f"✅ 图片生成成功！使用的种子值: {seed} | 耗时: {elapsed_time:.2f} 秒"
        
        return image, status_msg, gr.update(choices=related_prompts, value=related_prompts[0])
    
    except Exception as e:
        return None, f"❌ 生成失败: {str(e)}", gr.update()

def send_to_edit(image):
    """将图片发送到编辑标签页"""
    if image is None:
        return None, "没有可发送的图片！"
    return image, "✅ 图片已发送到编辑标签页！"

def edit_image(image, prompt, height, width, guidance_scale, num_steps, seed):
    """图片编辑函数 - 基于新提示词重新生成"""
    if not prompt.strip():
        return None, "请输入编辑提示词！"
    
    try:
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        start_time = time.time()
        
        # 使用新的提示词生成图片
        edited_image = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps),
            generator=generator
        ).images[0]
        
        elapsed_time = time.time() - start_time
        
        status_msg = f"✅ 图片编辑成功！使用的种子值: {seed} | 耗时: {elapsed_time:.2f} 秒"
        
        return edited_image, status_msg
    
    except Exception as e:
        return None, f"❌ 编辑失败: {str(e)}"

def use_example_prompt(example):
    """使用示例提示词"""
    return example

# 创建 Gradio 界面
with gr.Blocks(title="FLUX.2-Klein 图像生成器", theme=gr.themes.Soft()) as demo:
    # 顶部 YouTube 频道信息
    gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2em;">🎨 FLUX.2-Klein 图像生成器</h1>
            <p style="color: #f0f0f0; margin: 10px 0;">
                📺 <strong>AI 技术分享频道</strong> | 
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="color: #ffeb3b; text-decoration: none;">
                    🔗 订阅我的 YouTube 频道
                </a>
            </p>
        </div>
    """)
    
    with gr.Tabs():
        # 第一个标签页：文生图
        with gr.TabItem("🖼️ 文生图"):
            with gr.Row():
                with gr.Column(scale=1):
                    txt2img_prompt = gr.Textbox(
                        label="提示词",
                        placeholder="请输入您想要生成的图片描述...",
                        lines=3
                    )
                    
                    with gr.Row():
                        txt2img_height = gr.Slider(
                            minimum=256, maximum=2048, value=1024, step=64,
                            label="图片高度"
                        )
                        txt2img_width = gr.Slider(
                            minimum=256, maximum=2048, value=1024, step=64,
                            label="图片宽度"
                        )
                    
                    with gr.Row():
                        txt2img_guidance = gr.Slider(
                            minimum=0.0, maximum=10.0, value=1.0, step=0.1,
                            label="引导系数 (Guidance Scale)"
                        )
                        txt2img_steps = gr.Slider(
                            minimum=1, maximum=50, value=4, step=1,
                            label="推理步数"
                        )
                    
                    txt2img_seed = gr.Number(
                        label="随机种子 (-1 表示随机)",
                        value=-1,
                        precision=0
                    )
                    
                    with gr.Row():
                        txt2img_generate_btn = gr.Button("🎨 生成图片", variant="primary", size="lg")
                        txt2img_send_btn = gr.Button("📤 发送到编辑", variant="secondary", size="lg")
                    
                    txt2img_status = gr.Textbox(label="状态", interactive=False)
                    
                    # 示例提示词
                    gr.Markdown("### 📝 示例提示词 (点击使用)")
                    txt2img_examples = gr.Examples(
                        examples=[[p] for p in txt2img_example_prompts],
                        inputs=[txt2img_prompt],
                        label=""
                    )
                
                with gr.Column(scale=1):
                    txt2img_output = gr.Image(label="生成的图片", type="pil")
        
        # 第二个标签页：图片编辑
        with gr.TabItem("✏️ 图片编辑"):
            with gr.Row():
                with gr.Column(scale=1):
                    edit_input_image = gr.Image(
                        label="待编辑的图片",
                        type="pil"
                    )
                    
                    edit_prompt = gr.Textbox(
                        label="编辑提示词",
                        placeholder="请输入编辑描述...",
                        lines=3
                    )
                    
                    # 相关编辑提示词下拉菜单
                    edit_related_prompts = gr.Dropdown(
                        label="相关编辑提示词 (基于文生图提示词)",
                        choices=[],
                        interactive=True
                    )
                    
                    with gr.Row():
                        edit_height = gr.Slider(
                            minimum=256, maximum=2048, value=1024, step=64,
                            label="图片高度"
                        )
                        edit_width = gr.Slider(
                            minimum=256, maximum=2048, value=1024, step=64,
                            label="图片宽度"
                        )
                    
                    with gr.Row():
                        edit_guidance = gr.Slider(
                            minimum=0.0, maximum=10.0, value=1.0, step=0.1,
                            label="引导系数 (Guidance Scale)"
                        )
                        edit_steps = gr.Slider(
                            minimum=1, maximum=50, value=4, step=1,
                            label="推理步数"
                        )
                    
                    edit_seed = gr.Number(
                        label="随机种子 (-1 表示随机)",
                        value=-1,
                        precision=0
                    )
                    
                    edit_btn = gr.Button("✨ 应用编辑", variant="primary", size="lg")
                    edit_status = gr.Textbox(label="状态", interactive=False)
                
                with gr.Column(scale=1):
                    edit_output = gr.Image(label="编辑后的图片", type="pil")
    
    # 底部信息
    gr.HTML("""
        <div style="text-align: center; padding: 15px; margin-top: 20px; color: #666;">
            <p>💡 提示：在「文生图」标签页生成图片后，点击「发送到编辑」可以将图片发送到编辑标签页进行进一步编辑。</p>
            <p>📺 更多 AI 技术内容请关注: <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">AI 技术分享频道</a></p>
        </div>
    """)
    
    # 事件绑定
    txt2img_generate_btn.click(
        fn=generate_image,
        inputs=[txt2img_prompt, txt2img_height, txt2img_width, txt2img_guidance, txt2img_steps, txt2img_seed],
        outputs=[txt2img_output, txt2img_status, edit_related_prompts]
    )
    
    txt2img_send_btn.click(
        fn=send_to_edit,
        inputs=[txt2img_output],
        outputs=[edit_input_image, txt2img_status]
    )
    
    # 选择相关提示词时更新编辑提示词
    edit_related_prompts.change(
        fn=lambda x: x,
        inputs=[edit_related_prompts],
        outputs=[edit_prompt]
    )
    
    edit_btn.click(
        fn=edit_image,
        inputs=[edit_input_image, edit_prompt, edit_height, edit_width, edit_guidance, edit_steps, edit_seed],
        outputs=[edit_output, edit_status]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
