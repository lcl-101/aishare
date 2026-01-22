import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import gradio as gr
import os
import urllib.request
from PIL import Image

# 下载示例图片
EXAMPLE_IMAGE_URL = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
EXAMPLE_IMAGE_PATH = "example_image.jpeg"

def download_example_image():
    """下载示例图片到本地"""
    if not os.path.exists(EXAMPLE_IMAGE_PATH):
        print(f"正在下载示例图片...")
        try:
            urllib.request.urlretrieve(EXAMPLE_IMAGE_URL, EXAMPLE_IMAGE_PATH)
            print(f"示例图片已下载到: {EXAMPLE_IMAGE_PATH}")
        except Exception as e:
            print(f"下载示例图片失败: {e}")
    else:
        print(f"示例图片已存在: {EXAMPLE_IMAGE_PATH}")

# 初始化设备和数据类型
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == "mps" else torch.bfloat16

print(f"使用设备: {device}")
print(f"使用数据类型: {dtype}")

# 加载本地模型
MODEL_PATH = "checkpoints/LightOnOCR-2-1B"
print(f"正在加载模型: {MODEL_PATH}")
model = LightOnOcrForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
processor = LightOnOcrProcessor.from_pretrained(MODEL_PATH)
print("模型加载完成")

def ocr_inference(image, prompt, max_tokens):
    """
    对图片进行 OCR 识别
    
    Args:
        image: PIL Image 或图片路径
        prompt: 用户提示词（可选）
        max_tokens: 最大生成 token 数量
    
    Returns:
        识别的文本结果
    """
    if image is None:
        return "请上传图片或选择示例图片"
    
    try:
        # 如果是路径字符串，加载图片
        if isinstance(image, str):
            image = Image.open(image)
        
        # 保存临时图片并获取路径
        temp_image_path = "/tmp/temp_ocr_image.png"
        image.save(temp_image_path)
        
        # 构建对话格式，使用本地文件路径
        content = [{"type": "image", "url": temp_image_path}]
        
        # 如果有提示词，添加到内容中
        if prompt and prompt.strip():
            content.append({"type": "text", "text": prompt.strip()})
        
        conversation = [{"role": "user", "content": content}]
        
        # 应用聊天模板并处理输入
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # 将输入移动到正确的设备
        inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
        
        # 生成输出
        output_ids = model.generate(**inputs, max_new_tokens=int(max_tokens))
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        return output_text
        
    except Exception as e:
        return f"识别过程中出现错误: {str(e)}"

# 下载示例图片
download_example_image()

# 创建 Gradio 界面
with gr.Blocks(title="LightOnOCR-2 文字识别") as demo:
    # YouTube 频道信息
    gr.Markdown(
        """
        # LightOnOCR-2 文字识别系统
        
        ### 📺 欢迎关注 [AI 技术分享频道](https://www.youtube.com/@rongyikanshijie-ai)
        
        ---
        """
    )
    
    gr.Markdown(
        """
        ## 使用说明
        
        本系统使用 LightOnOCR-2-1B 模型进行光学字符识别（OCR）。上传图片或选择示例图片，系统将自动识别图片中的文字内容。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 图片输入
            image_input = gr.Image(
                label="上传图片",
                type="pil",
                sources=["upload", "clipboard"]
            )
            
            # 提示词输入
            prompt_input = gr.Textbox(
                label="提示词（可选）",
                placeholder="例如：Convert to markdown / Extract text / Describe the image",
                lines=2,
                value=""
            )
            
            # 示例提示词
            gr.Markdown("**常用提示词示例：**")
            with gr.Row():
                example_prompts = [
                    gr.Button("Convert to markdown", size="sm"),
                    gr.Button("Extract all text", size="sm"),
                    gr.Button("Read this document", size="sm"),
                ]
            
            # 参数设置
            max_tokens_slider = gr.Slider(
                minimum=128,
                maximum=2048,
                value=1024,
                step=128,
                label="最大生成 Token 数量",
                info="控制输出文本的最大长度"
            )
            
            # 识别按钮
            submit_btn = gr.Button("开始识别", variant="primary", size="lg")
            
            # 清除按钮
            clear_btn = gr.Button("清除", size="sm")
            
        with gr.Column(scale=1):
            # 输出文本框
            output_text = gr.Textbox(
                label="识别结果",
                lines=20,
                placeholder="识别结果将在这里显示..."
            )
    
    # 示例图片
    gr.Markdown("### 📷 示例图片")
    gr.Examples(
        examples=[[EXAMPLE_IMAGE_PATH, "", 1024]] if os.path.exists(EXAMPLE_IMAGE_PATH) else [],
        inputs=[image_input, prompt_input, max_tokens_slider],
        outputs=output_text,
        fn=ocr_inference,
        cache_examples=False,
        label="点击加载示例"
    )
    
    # 绑定示例提示词按钮事件
    example_prompts[0].click(fn=lambda: "Convert to markdown", outputs=prompt_input)
    example_prompts[1].click(fn=lambda: "Extract all text", outputs=prompt_input)
    example_prompts[2].click(fn=lambda: "Read this document", outputs=prompt_input)
    
    # 页脚信息
    gr.Markdown(
        """
        ---
        
        **模型信息**: LightOnOCR-2-1B | **设备**: {} | **精度**: {}
        """.format(device, dtype)
    )
    
    # 事件绑定
    submit_btn.click(
        fn=ocr_inference,
        inputs=[image_input, prompt_input, max_tokens_slider],
        outputs=output_text
    )
    
    clear_btn.click(
        fn=lambda: (None, "", ""),
        inputs=None,
        outputs=[image_input, prompt_input, output_text]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
