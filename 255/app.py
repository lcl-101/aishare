import torch
import gradio as gr
from diffusers import Step1XEditPipelineV1P2
from PIL import Image

# 加载模型
print("=== 加载模型 ===")
pipe = Step1XEditPipelineV1P2.from_pretrained("checkpoints/Step1X-Edit-v1p2", torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("=== 模型加载完成 ===")

# 定义示例
EXAMPLES = [
    # [图片路径, 编辑提示词]
    ["checkpoints/Step1X-Edit-v1p2/examples/0000.jpg", "add a ruby pendant on the girl's neck"],
    ["checkpoints/Step1X-Edit-v1p2/examples/0001.png", "change the blazer color from red to blue"],
    ["checkpoints/Step1X-Edit-v1p2/examples/0002.jpg", "change the guitar color from blue to red"],
    ["checkpoints/Step1X-Edit-v1p2/examples/0003.png", "change the headband color to red"],
    ["checkpoints/Step1X-Edit-v1p2/examples/0004.jpg", "change the text 'NEW ENGLAND' to 'SWITZERLAND'"],
]


def process_image(
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    seed: int,
    enable_thinking_mode: bool,
    enable_reflection_mode: bool,
):
    """处理图片编辑请求"""
    if image is None:
        return None, "请上传图片"
    
    if not prompt.strip():
        return None, "请输入编辑提示词"
    
    # 转换图片格式
    image = image.convert("RGB")
    
    # 运行 pipeline
    pipe_output = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        generator=torch.Generator().manual_seed(seed),
        enable_thinking_mode=enable_thinking_mode,
        enable_reflection_mode=enable_reflection_mode,
    )
    
    # 构建输出信息
    info_text = ""
    if enable_thinking_mode and pipe_output.reformat_prompt:
        info_text += f"**Reformat Prompt:** {pipe_output.reformat_prompt}\n\n"
    
    if enable_reflection_mode and pipe_output.think_info:
        info_text += f"**Think Info:** {pipe_output.think_info[0]}\n\n"
    
    if enable_reflection_mode and pipe_output.best_info:
        info_text += f"**Best Info:** {pipe_output.best_info[0]}\n\n"
    
    # 返回最终图片和信息
    final_image = pipe_output.final_images[0] if pipe_output.final_images else pipe_output.images[0]
    
    return final_image, info_text if info_text else "编辑完成"


# 创建 Gradio 界面
with gr.Blocks(title="Step1X-Edit V1P2") as demo:
    gr.Markdown(
        """
        # 🎨 Step1X-Edit V1P2
        
        基于 Step1X-Edit 的图像编辑工具，支持通过自然语言指令编辑图片。
        
        **使用方法：**
        1. 上传图片或选择下方示例
        2. 输入编辑提示词（如：添加项链、改变颜色、替换文字等）
        3. 调整参数后点击"开始编辑"
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 输入区域
            input_image = gr.Image(label="输入图片", type="pil", height=400)
            prompt = gr.Textbox(
                label="编辑提示词",
                placeholder="例如：add a ruby pendant on the girl's neck",
                lines=2
            )
            
            with gr.Accordion("高级设置", open=False):
                num_steps = gr.Slider(
                    minimum=10, maximum=100, value=50, step=1,
                    label="推理步数 (num_inference_steps)"
                )
                cfg_scale = gr.Slider(
                    minimum=1.0, maximum=15.0, value=6.0, step=0.5,
                    label="CFG Scale (true_cfg_scale)"
                )
                seed = gr.Number(value=42, label="随机种子", precision=0)
                enable_thinking = gr.Checkbox(value=True, label="启用思考模式 (Thinking Mode)")
                enable_reflection = gr.Checkbox(value=True, label="启用反思模式 (Reflection Mode)")
            
            submit_btn = gr.Button("🚀 开始编辑", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # 输出区域
            output_image = gr.Image(label="编辑结果", type="pil", height=400)
            output_info = gr.Markdown(label="处理信息")
    
    # 示例区域
    gr.Markdown("### 📷 示例图片")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_image, prompt],
        label="点击选择示例",
        examples_per_page=5,
    )
    
    # 绑定事件
    submit_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            prompt,
            num_steps,
            cfg_scale,
            seed,
            enable_thinking,
            enable_reflection,
        ],
        outputs=[output_image, output_info],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
