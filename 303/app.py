import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests
import os
from io import BytesIO

# ==================== 配置 ====================
MODEL_PATH = "checkpoints/translategemma-27b-it"
EXAMPLE_IMAGE_DIR = "example_images"
EXAMPLE_IMAGE_URL = "https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg"
EXAMPLE_IMAGE_PATH = os.path.join(EXAMPLE_IMAGE_DIR, "example_traffic_sign.jpg")

# ==================== 下载示例图片 ====================
def download_example_image():
    """下载官方示例图片"""
    if not os.path.exists(EXAMPLE_IMAGE_DIR):
        os.makedirs(EXAMPLE_IMAGE_DIR)
    
    if not os.path.exists(EXAMPLE_IMAGE_PATH):
        print(f"正在下载示例图片: {EXAMPLE_IMAGE_URL}")
        try:
            response = requests.get(EXAMPLE_IMAGE_URL, timeout=30)
            response.raise_for_status()
            with open(EXAMPLE_IMAGE_PATH, 'wb') as f:
                f.write(response.content)
            print(f"示例图片已保存到: {EXAMPLE_IMAGE_PATH}")
        except Exception as e:
            print(f"下载示例图片失败: {e}")
            return None
    return EXAMPLE_IMAGE_PATH

# ==================== 加载模型 ====================
print("=" * 50)
print("正在加载 TranslateGemma 模型...")
print(f"模型路径: {MODEL_PATH}")
print("=" * 50)

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, 
    device_map="auto",
    dtype=torch.bfloat16
)
print("模型加载完成！")

# 下载示例图片
example_image_path = download_example_image()

# ==================== 翻译函数 ====================
def translate_text(text, source_lang, target_lang):
    """文本翻译"""
    if not text.strip():
        return "请输入要翻译的文本"
    
    # 设置语言代码
    lang_map = {
        "中文": "zh",
        "英文": "en"
    }
    source_code = lang_map.get(source_lang, "zh")
    target_code = lang_map.get(target_lang, "en")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_code,
                    "target_lang_code": target_code,
                    "text": text,
                }
            ],
        }
    ]
    
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, 
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = len(inputs['input_ids'][0])
        
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded
    except Exception as e:
        return f"翻译出错: {str(e)}"

def translate_image(image, source_lang, target_lang):
    """图片文字提取与翻译"""
    if image is None:
        return "请上传图片"
    
    # 设置语言代码
    lang_map = {
        "中文": "zh",
        "英文": "en",
        "捷克语": "cs",
        "德语": "de",
        "法语": "fr",
        "日语": "ja",
        "韩语": "ko",
        "西班牙语": "es"
    }
    source_code = lang_map.get(source_lang, "en")
    target_code = lang_map.get(target_lang, "zh")
    
    # 保存图片到临时文件
    temp_image_path = "temp_upload_image.jpg"
    if isinstance(image, str):
        temp_image_path = image
    else:
        image.save(temp_image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_lang_code": source_code,
                    "target_lang_code": target_code,
                    "url": temp_image_path,
                },
            ],
        }
    ]
    
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, 
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = len(inputs['input_ids'][0])
        
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded
    except Exception as e:
        return f"翻译出错: {str(e)}"

# ==================== 示例文本 ====================
text_examples_zh_to_en = [
    ["人工智能正在改变我们的生活方式。", "中文", "英文"],
    ["今天天气真好，我们一起去公园散步吧。", "中文", "英文"],
    ["这款产品采用了最新的技术，性能非常出色。", "中文", "英文"],
    ["学习一门新语言需要时间和耐心。", "中文", "英文"],
    ["中国是一个拥有悠久历史和灿烂文化的国家。", "中文", "英文"],
]

text_examples_en_to_zh = [
    ["Artificial intelligence is transforming the way we live.", "英文", "中文"],
    ["The weather is beautiful today, let's take a walk in the park.", "英文", "中文"],
    ["This product uses the latest technology and has excellent performance.", "英文", "中文"],
    ["Learning a new language takes time and patience.", "英文", "中文"],
    ["Machine learning models are becoming increasingly powerful.", "英文", "中文"],
]

# ==================== Gradio 界面 ====================
# 自定义 CSS
custom_css = """
.youtube-banner {
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    padding: 15px 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}
.youtube-banner a {
    color: white !important;
    text-decoration: none;
    font-size: 18px;
    font-weight: bold;
}
.youtube-banner a:hover {
    text-decoration: underline;
}
"""

with gr.Blocks(title="TranslateGemma 中英互译") as demo:
    # YouTube 频道信息
    gr.HTML("""
    <div class="youtube-banner">
        <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">
            📺 欢迎订阅我的 YouTube 频道: AI 技术分享频道
        </a>
    </div>
    """)
    
    gr.Markdown("""
    # 🌐 TranslateGemma 中英互译系统
    
    基于 Google TranslateGemma-27B 模型，支持中英文文本互译和图片文字提取翻译。
    
    **模型特点：**
    - 支持 55 种语言翻译
    - 支持图片文字提取与翻译
    - 轻量级、高性能
    """)
    
    with gr.Tabs():
        # ==================== 文本翻译 Tab ====================
        with gr.TabItem("📝 文本翻译"):
            gr.Markdown("### 输入文本进行中英互译")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="输入文本",
                        placeholder="请输入要翻译的文本...",
                        lines=5
                    )
                    with gr.Row():
                        source_lang_text = gr.Dropdown(
                            choices=["中文", "英文"],
                            value="中文",
                            label="源语言"
                        )
                        target_lang_text = gr.Dropdown(
                            choices=["中文", "英文"],
                            value="英文",
                            label="目标语言"
                        )
                    translate_text_btn = gr.Button("🚀 开始翻译", variant="primary")
                
                with gr.Column():
                    text_output = gr.Textbox(
                        label="翻译结果",
                        lines=5,
                        interactive=False
                    )
            
            gr.Markdown("### 📚 中文 → 英文 示例")
            gr.Examples(
                examples=text_examples_zh_to_en,
                inputs=[text_input, source_lang_text, target_lang_text],
                label=""
            )
            
            gr.Markdown("### 📚 英文 → 中文 示例")
            gr.Examples(
                examples=text_examples_en_to_zh,
                inputs=[text_input, source_lang_text, target_lang_text],
                label=""
            )
            
            translate_text_btn.click(
                fn=translate_text,
                inputs=[text_input, source_lang_text, target_lang_text],
                outputs=text_output
            )
        
        # ==================== 图片翻译 Tab ====================
        with gr.TabItem("🖼️ 图片文字提取与翻译"):
            gr.Markdown("### 上传图片，提取并翻译图片中的文字")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="上传图片",
                        type="pil"
                    )
                    with gr.Row():
                        source_lang_image = gr.Dropdown(
                            choices=["中文", "英文", "捷克语", "德语", "法语", "日语", "韩语", "西班牙语"],
                            value="英文",
                            label="图片文字语言"
                        )
                        target_lang_image = gr.Dropdown(
                            choices=["中文", "英文", "捷克语", "德语", "法语", "日语", "韩语", "西班牙语"],
                            value="中文",
                            label="翻译目标语言"
                        )
                    translate_image_btn = gr.Button("🚀 提取并翻译", variant="primary")
                
                with gr.Column():
                    image_output = gr.Textbox(
                        label="翻译结果",
                        lines=8,
                        interactive=False
                    )
            
            # 图片示例
            if example_image_path and os.path.exists(example_image_path):
                gr.Markdown("### 📷 示例图片（捷克语交通标志）")
                gr.Examples(
                    examples=[[example_image_path, "捷克语", "中文"]],
                    inputs=[image_input, source_lang_image, target_lang_image],
                    label=""
                )
            
            translate_image_btn.click(
                fn=translate_image,
                inputs=[image_input, source_lang_image, target_lang_image],
                outputs=image_output
            )
    
    gr.Markdown("""
    ---
    **说明：**
    - 本系统基于 Google TranslateGemma-27B-IT 模型
    - 当前仅支持中英文互译
    - 图片翻译会自动提取图片中的文字并翻译
    """)

# ==================== 启动应用 ====================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css, theme=gr.themes.Soft())
