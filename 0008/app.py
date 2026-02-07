"""
GLM-OCR Web 应用程序
基于 Gradio 的文档识别系统
"""

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
import tempfile
from pdf2image import convert_from_path
import io

# 模型路径
MODEL_PATH = "checkpoints/GLM-OCR"

# 预定义的提示词模板
PROMPT_TEMPLATES = {
    "文字识别": "Text Recognition:",
    "公式识别": "Formula Recognition:",
    "表格识别": "Table Recognition:",
    "自定义": ""
}

# 信息提取示例 JSON 模板
INFO_EXTRACTION_EXAMPLE = '''请按下列JSON格式输出图中信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}'''

# 全局变量存储模型和处理器
model = None
processor = None


def load_model():
    """加载模型和处理器"""
    global model, processor
    if model is None or processor is None:
        print("正在加载模型...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )
        print("模型加载完成！")
    return model, processor


def clean_latex_output(text):
    """清理 LaTeX 输出，移除多余的特殊标记"""
    import re
    
    # 移除特殊标记
    special_tokens = [
        "<|endoftext|>", "<|user|>", "<|assistant|>", 
        "<|system|>", "<|end|>", "<|im_end|>", "<|im_start|>"
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    
    # 清理首尾空白
    text = text.strip()
    
    return text


def normalize_latex(text):
    """
    规范化 LaTeX 格式，去除多余的空格
    例如: q _ {\sigma} -> q_{\sigma}
          \boldsymbol {x} -> \boldsymbol{x}
    但保留主表达式中运算符周围的空格
    """
    import re
    
    # 去除下划线前后的空格: x _ {y} -> x_{y}
    text = re.sub(r'\s*_\s*', '_', text)
    
    # 去除上标前后的空格: x ^ {2} -> x^{2}
    text = re.sub(r'\s*\^\s*', '^', text)
    
    # 去除 LaTeX 命令和花括号之间的空格: \cmd {arg} -> \cmd{arg}
    text = re.sub(r'(\\[a-zA-Z]+)\s+\{', r'\1{', text)
    
    # 去除 \left 和括号之间的空格: \left ( -> \left(
    text = re.sub(r'\\left\s*([(\[{|])', r'\\left\1', text)
    
    # 去除 \right 和括号之间的空格: \right ) -> \right)
    text = re.sub(r'\\right\s*([)\]}|])', r'\\right\1', text)
    
    # 只在下标/上标的花括号内去除减号周围的空格
    # _{t - 1} -> _{t-1}  和  ^{t - 1} -> ^{t-1}
    text = re.sub(r'_\{([a-zA-Z0-9]+)\s*-\s*([a-zA-Z0-9]+)\}', r'_{\1-\2}', text)
    text = re.sub(r'\^\{([a-zA-Z0-9]+)\s*-\s*([a-zA-Z0-9]+)\}', r'^{\1-\2}', text)
    
    # 规范化多个连续空格为单个空格
    text = re.sub(r'  +', ' ', text)
    
    return text


def format_as_latex(text, is_formula=False):
    """
    格式化为标准 LaTeX 格式
    - 如果是公式，确保有正确的数学环境包裹
    - 清理多余的标记
    """
    # 首先清理特殊标记
    text = clean_latex_output(text)
    
    if not is_formula:
        return text
    
    # 检查是否已经有数学环境包裹
    text_stripped = text.strip()
    
    # 如果已经有 $ 或 $$ 包裹，保持不变
    if (text_stripped.startswith('$') and text_stripped.endswith('$')) or \
       (text_stripped.startswith('$$') and text_stripped.endswith('$$')) or \
       (text_stripped.startswith('\\[') and text_stripped.endswith('\\]')) or \
       (text_stripped.startswith('\\begin{') and text_stripped.endswith('\\end{')):
        return text
    
    return text


def recognize_image(image, prompt_text):
    """识别单张图片"""
    model, processor = load_model()
    
    # 如果图片是文件路径，则加载它
    if isinstance(image, str):
        image = Image.open(image)
    
    # 确保图片是 RGB 模式
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 保存临时图片
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        temp_path = tmp.name
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": temp_path
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ],
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=False
        )
        
        # 清理特殊标记（使用新的清理函数）
        output_text = clean_latex_output(output_text)
        
        return output_text
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_single_image(image, prompt_type, custom_prompt):
    """处理单张图片"""
    if image is None:
        return None, "", "", ""
    
    # 确定使用的提示词
    if prompt_type == "自定义":
        prompt_text = custom_prompt if custom_prompt else "Text Recognition:"
    else:
        prompt_text = PROMPT_TEMPLATES.get(prompt_type, "Text Recognition:")
    
    try:
        result = recognize_image(image, prompt_text)
        
        # 如果是公式识别，规范化 LaTeX 格式
        latex_result = result
        rendered_latex = ""
        if prompt_type == "公式识别":
            # 规范化 LaTeX 格式（去除多余空格）
            result = normalize_latex(result)
            
            # 提供纯 LaTeX 格式（去掉 $ 符号）
            latex_result = result.strip()
            if latex_result.startswith('$$') and latex_result.endswith('$$'):
                latex_result = latex_result[2:-2].strip()
            elif latex_result.startswith('$') and latex_result.endswith('$'):
                latex_result = latex_result[1:-1].strip()
            
            # 渲染预览（使用 $$ 包裹以居中显示）
            rendered_latex = f"$$\n{latex_result}\n$$"
        
        return image, result, latex_result, rendered_latex
    except Exception as e:
        return image, f"识别出错: {str(e)}", "", ""


def convert_pdf_to_images(pdf_path):
    """将 PDF 转换为图片列表"""
    try:
        images = convert_from_path(pdf_path, dpi=200)
        return images
    except Exception as e:
        raise Exception(f"PDF 转换失败: {str(e)}")


def process_pdf(pdf_file, prompt_type, custom_prompt, progress=gr.Progress()):
    """处理 PDF 文件"""
    if pdf_file is None:
        return [], "请先上传 PDF 文件！"
    
    # 确定使用的提示词
    if prompt_type == "自定义":
        prompt_text = custom_prompt if custom_prompt else "Text Recognition:"
    else:
        prompt_text = PROMPT_TEMPLATES.get(prompt_type, "Text Recognition:")
    
    try:
        # 转换 PDF 为图片
        progress(0, desc="正在转换 PDF...")
        images = convert_pdf_to_images(pdf_file)
        
        results = []
        total_pages = len(images)
        
        for i, img in enumerate(images):
            progress((i + 1) / total_pages, desc=f"正在识别第 {i + 1}/{total_pages} 页...")
            
            try:
                result_text = recognize_image(img, prompt_text)
            except Exception as e:
                result_text = f"识别出错: {str(e)}"
            
            results.append((img, f"第 {i + 1} 页识别结果:\n\n{result_text}"))
        
        return results, f"✅ 成功处理 {total_pages} 页"
    except Exception as e:
        return [], f"❌ 处理失败: {str(e)}"


def update_custom_prompt_visibility(prompt_type):
    """根据提示词类型更新自定义提示词框的可见性"""
    return gr.update(visible=(prompt_type == "自定义"))


def fill_info_extraction_template():
    """填充信息提取模板"""
    return INFO_EXTRACTION_EXAMPLE


# 自定义 CSS
CUSTOM_CSS = """
.youtube-banner {
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    padding: 15px 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}
.youtube-banner a {
    color: white;
    text-decoration: none;
    font-size: 18px;
    font-weight: bold;
}
.youtube-banner a:hover {
    text-decoration: underline;
}
.result-gallery {
    min-height: 400px;
}
"""

# 创建 Gradio 界面
def create_interface():
    with gr.Blocks(
        title="GLM-OCR 文档识别系统"
    ) as demo:
        # YouTube 频道信息横幅
        gr.HTML("""
        <div class="youtube-banner">
            <a href="https://www.youtube.com/@rongyi-ai" target="_blank">
                📺 欢迎访问我的 YouTube 频道：AI 技术分享频道 ➜ https://www.youtube.com/@rongyi-ai
            </a>
        </div>
        """)
        
        gr.Markdown("""
        # 🔍 GLM-OCR 文档识别系统
        
        上传图片或 PDF 文件，使用 GLM-OCR 模型进行文档识别。支持文字识别、公式识别、表格识别以及自定义信息提取。
        """)
        
        with gr.Tabs():
            # 单张图片识别标签页
            with gr.TabItem("📷 单张图片识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="上传图片",
                            type="pil",
                            height=400
                        )
                        
                        prompt_type_image = gr.Radio(
                            choices=list(PROMPT_TEMPLATES.keys()),
                            value="文字识别",
                            label="选择识别类型"
                        )
                        
                        custom_prompt_image = gr.Textbox(
                            label="自定义提示词",
                            placeholder="输入自定义提示词...",
                            lines=5,
                            visible=False
                        )
                        
                        with gr.Row():
                            fill_template_btn_image = gr.Button(
                                "📋 填充信息提取模板", 
                                size="sm"
                            )
                        
                        recognize_btn = gr.Button(
                            "🚀 开始识别", 
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        image_output = gr.Image(
                            label="原始图片",
                            height=300
                        )
                        result_output = gr.Textbox(
                            label="识别结果",
                            lines=10
                        )
                        latex_output = gr.Textbox(
                            label="📋 纯 LaTeX 公式 (公式识别时去掉 $ 符号，可直接复制)",
                            lines=5,
                            placeholder="公式识别结果会显示在这里..."
                        )
                        latex_preview = gr.Markdown(
                            label="📐 公式渲染预览",
                            value=""
                        )
                
                # 事件绑定
                prompt_type_image.change(
                    fn=update_custom_prompt_visibility,
                    inputs=prompt_type_image,
                    outputs=custom_prompt_image
                )
                
                fill_template_btn_image.click(
                    fn=fill_info_extraction_template,
                    outputs=custom_prompt_image
                )
                
                recognize_btn.click(
                    fn=process_single_image,
                    inputs=[image_input, prompt_type_image, custom_prompt_image],
                    outputs=[image_output, result_output, latex_output, latex_preview]
                )
            
            # PDF 文件识别标签页
            with gr.TabItem("📄 PDF 文件识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_input = gr.File(
                            label="上传 PDF 文件",
                            file_types=[".pdf"],
                            type="filepath"
                        )
                        
                        prompt_type_pdf = gr.Radio(
                            choices=list(PROMPT_TEMPLATES.keys()),
                            value="文字识别",
                            label="选择识别类型"
                        )
                        
                        custom_prompt_pdf = gr.Textbox(
                            label="自定义提示词",
                            placeholder="输入自定义提示词...",
                            lines=5,
                            visible=False
                        )
                        
                        with gr.Row():
                            fill_template_btn_pdf = gr.Button(
                                "📋 填充信息提取模板", 
                                size="sm"
                            )
                        
                        process_pdf_btn = gr.Button(
                            "🚀 开始处理 PDF", 
                            variant="primary",
                            size="lg"
                        )
                        
                        pdf_status = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        # 使用 Gallery 显示所有页面的识别结果
                        pdf_gallery = gr.Gallery(
                            label="PDF 页面预览",
                            show_label=True,
                            columns=2,
                            rows=2,
                            height=400,
                            object_fit="contain"
                        )
                
                # PDF 识别结果详情
                gr.Markdown("### 📝 各页识别结果详情")
                pdf_results_display = gr.Dataframe(
                    headers=["页码", "识别结果"],
                    datatype=["str", "str"],
                    col_count=(2, "fixed"),
                    wrap=True,
                    visible=False
                )
                
                # 使用 Accordion 显示每页详细结果
                pdf_detail_output = gr.Markdown(
                    label="详细结果",
                    visible=True
                )
                
                # 事件绑定
                prompt_type_pdf.change(
                    fn=update_custom_prompt_visibility,
                    inputs=prompt_type_pdf,
                    outputs=custom_prompt_pdf
                )
                
                fill_template_btn_pdf.click(
                    fn=fill_info_extraction_template,
                    outputs=custom_prompt_pdf
                )
                
                def process_pdf_and_display(pdf_file, prompt_type, custom_prompt, progress=gr.Progress()):
                    """处理 PDF 并格式化显示结果"""
                    results, status = process_pdf(pdf_file, prompt_type, custom_prompt, progress)
                    
                    if not results:
                        return [], status, ""
                    
                    # 准备 Gallery 显示的图片
                    gallery_images = [img for img, _ in results]
                    
                    # 格式化详细结果
                    detail_md = ""
                    for i, (img, result) in enumerate(results):
                        detail_md += f"---\n\n## 📄 第 {i + 1} 页\n\n"
                        detail_md += f"```\n{result.replace(f'第 {i + 1} 页识别结果:', '').strip()}\n```\n\n"
                    
                    return gallery_images, status, detail_md
                
                process_pdf_btn.click(
                    fn=process_pdf_and_display,
                    inputs=[pdf_input, prompt_type_pdf, custom_prompt_pdf],
                    outputs=[pdf_gallery, pdf_status, pdf_detail_output]
                )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 支持的识别类型
            
            **1. 文档解析 (Document Parsing)**
            - **文字识别**: 提取文档中的文字内容
            - **公式识别**: 识别数学公式
            - **表格识别**: 提取表格结构和内容
            
            **2. 信息提取 (Information Extraction)**
            - 使用 **自定义** 选项，按照 JSON 格式定义需要提取的字段
            - 点击 **填充信息提取模板** 按钮可以快速填入示例模板
            
            ### 提示词示例
            
            文档解析提示词:
            ```
            Text Recognition:
            Formula Recognition:
            Table Recognition:
            ```
            
            信息提取提示词示例:
            ```
            请按下列JSON格式输出图中信息:
            {
                "id_number": "",
                "last_name": "",
                "first_name": "",
                ...
            }
            ```
            
            ### 注意事项
            - PDF 文件会被转换为图片后逐页识别
            - 大型 PDF 文件处理可能需要较长时间
            - 使用信息提取时，输出会严格遵循定义的 JSON 格式
            """)
        
        gr.Markdown("""
        ---
        <center>
        💡 基于 GLM-OCR 模型 | 由 Gradio 提供支持
        </center>
        """)
    
    return demo


if __name__ == "__main__":
    # 预加载模型
    print("正在初始化应用...")
    load_model()
    
    # 创建并启动应用
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )
