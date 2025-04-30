import gradio as gr
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxTransformer2DModel
from transformers import T5EncoderModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from PIL import Image
import numpy as np
import os

# 全局变量，用于跟踪模型是否已加载
pipe = None
model_loaded = False

# --- 模型加载函数 ---
def load_model():
    global pipe, model_loaded
    
    # 如果模型已经加载，则直接返回
    if model_loaded and pipe is not None:
        return pipe, True
    
    print("开始加载模型...")
    base_model = 'checkpoints/FLUX.1-dev'
    controlnet_model_union = 'checkpoints/FLUX.1-dev-ControlNet-Union-Pro-2.0'

    # 检查模型路径
    if not os.path.exists(base_model) or not os.path.exists(controlnet_model_union):
        print(f"错误：模型路径不存在，请确保 '{base_model}' 和 '{controlnet_model_union}' 路径有效")
        return None, False

    # 4-bit 量化配置
    quant_config_text_encoder = TransformersBitsAndBytesConfig(load_in_4bit=True)
    quant_config_transformer = DiffusersBitsAndBytesConfig(load_in_4bit=True)

    # 加载模型组件
    try:
        # 加载 text_encoder_2（使用4-bit量化）
        text_encoder_2_4bit = T5EncoderModel.from_pretrained(
            base_model,
            subfolder="text_encoder_2",
            quantization_config=quant_config_text_encoder,
            torch_dtype=torch.float16
        )
        
        # 加载 transformer（使用4-bit量化）
        transformer_4bit = FluxTransformer2DModel.from_pretrained(
            base_model, 
            subfolder="transformer",
            quantization_config=quant_config_transformer,
            torch_dtype=torch.float16
        )
        
        # 加载 ControlNet 模型
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model_union, 
            torch_dtype=torch.float16
        )
        
        # 创建 pipeline
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            transformer=transformer_4bit,
            text_encoder_2=text_encoder_2_4bit,
            torch_dtype=torch.float16
        )
        pipe.to("cuda")
        model_loaded = True
        print("模型加载完成！")
        return pipe, True
    except Exception as e:
        print(f"模型加载出错: {e}")
        return None, False

# --- 推理函数 ---
def generate_image(control_image, prompt, progress=gr.Progress()):
    global pipe, model_loaded
    
    if control_image is None:
        raise gr.Error("请上传一张控制图片")
    
    if not prompt:
        raise gr.Error("请输入提示词")
    
    # 处理控制图像
    if isinstance(control_image, np.ndarray):
        control_image = Image.fromarray(control_image).convert("RGB")
    
    width, height = control_image.size
    print(f"控制图像尺寸: {width}x{height}")
    
    # 加载模型（如果尚未加载）
    progress(0.1, desc="检查模型状态...")
    if not model_loaded or pipe is None:
        progress(0.2, desc="开始加载模型...（首次生成需要等待较长时间）")
        pipe, load_success = load_model()
        if not load_success:
            raise gr.Error("模型加载失败，请检查控制台错误信息")
    
    print(f"开始生成图像，提示词: {prompt[:50]}...")
    progress(0.5, desc="正在生成图像...")
    
    # 执行推理
    try:
        generator = torch.Generator(device="cuda").manual_seed(42)
        result = pipe(
            prompt=prompt,
            control_image=control_image,
            width=width,
            height=height,
            controlnet_conditioning_scale=0.7,
            control_guidance_end=0.8,
            num_inference_steps=30,
            guidance_scale=3.5,
            generator=generator
        ).images[0]
        
        progress(1.0, desc="生成完成！")
        print("图像生成完成")
        return result
    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        raise gr.Error(f"生成失败: {e}")

# --- Gradio UI ---
# 示例数据
example_prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
example_image = "examples/canny.png"

# 第二个示例数据
robot_prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
robot_image = "examples/depth.png"

# 检查示例图片是否存在并创建示例列表
examples = []
if os.path.exists(example_image):
    examples.append([example_image, example_prompt])
else:
    print(f"警告: 示例图片 '{example_image}' 不存在")

if os.path.exists(robot_image):
    examples.append([robot_image, robot_prompt])
else:
    print(f"警告: 示例图片 '{robot_image}' 不存在")

# 界面样式
css = """
.container {
    max-width: 1200px;
    margin: auto;
}
#output-image, #input-image {
    height: 500px;
}
#submit-button {
    margin: 20px auto;
    max-width: 300px;
}
"""

# 创建 Gradio 界面
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FLUX ControlNet 演示 (4-bit 量化版)")
    gr.Markdown("上传一张控制图片 (如 Canny 边缘图)，输入提示词，点击生成按钮")
    gr.Markdown("**注意**: 首次点击生成按钮时会加载模型，可能需要等待一段时间")
    
    with gr.Row():
        # 左侧：上传控制图片
        input_image = gr.Image(
            type="pil", 
            label="控制图片", 
            elem_id="input-image"
        )
        
        # 右侧：显示结果图片
        output_image = gr.Image(
            type="pil", 
            label="生成结果", 
            elem_id="output-image"
        )
    
    # 提示词输入
    prompt = gr.Textbox(
        label="提示词", 
        lines=3, 
        placeholder="请输入提示词...",
        value=example_prompt
    )
    
    # 提交按钮
    submit_button = gr.Button(
        "生成图片", 
        variant="primary", 
        elem_id="submit-button"
    )
    
    # 示例区域
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[input_image, prompt],
            outputs=output_image,
            fn=generate_image,
            cache_examples=False,
            label="示例"
        )
    
    # 绑定提交按钮事件
    submit_button.click(
        fn=generate_image,
        inputs=[input_image, prompt],
        outputs=output_image
    )

# 启动应用
if __name__ == "__main__":
    print("应用已启动，模型将在首次生成时加载")
    demo.launch(server_name="0.0.0.0", share=False)
