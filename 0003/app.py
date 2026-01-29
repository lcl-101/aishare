"""
DeepSeek-OCR-2 Gradio Web 应用
支持图片和 PDF 文件的 OCR 解析
"""

import gradio as gr
import torch
import os
import tempfile
import shutil
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# PDF 处理库
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("警告: PyMuPDF 未安装，PDF 功能将不可用。请运行: pip install PyMuPDF")

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型路径
MODEL_PATH = "./checkpoints/DeepSeek-OCR-2"

# 全局模型和 tokenizer
model = None
tokenizer = None


def load_model():
    """加载模型"""
    global model, tokenizer
    if model is None:
        print("正在加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        model = model.eval().cuda().to(torch.bfloat16)
        print("模型加载完成！")
    return model, tokenizer


def pdf_to_images(pdf_path: str) -> list:
    """将 PDF 转换为图片列表"""
    if not PDF_SUPPORT:
        raise ValueError("PyMuPDF 未安装，无法处理 PDF 文件")

    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # 使用较高的 DPI 以获得更好的 OCR 效果
        mat = fitz.Matrix(2.0, 2.0)  # 2x 缩放
        pix = page.get_pixmap(matrix=mat)
        # 转换为 PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def process_single_image(image_path: str, prompt: str, output_dir: str) -> tuple:
    """处理单张图片"""
    global model, tokenizer
    model, tokenizer = load_model()

    # 调用模型推理
    model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=True,
    )

    # 读取结果文件
    result_file = os.path.join(output_dir, "result.mmd")
    result_image_file = os.path.join(output_dir, "result_with_boxes.jpg")

    result_text = ""
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            result_text = f.read()

    result_image = None
    if os.path.exists(result_image_file):
        # 加载图片到内存并复制，避免文件删除后数据丢失
        img = Image.open(result_image_file)
        result_image = img.copy()
        img.close()

    return result_text, result_image


def process_image(image, prompt_type: str, custom_prompt: str) -> tuple:
    """处理上传的图片"""
    if image is None:
        return "请上传图片", None

    # 确定使用的提示词
    prompt_map = {
        "文档转 Markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "图片 OCR": "<image>\n<|grounding|>OCR this image.",
        "纯文字提取（无布局）": "<image>\nFree OCR.",
        "图表解析": "<image>\nParse the figure.",
        "图片详细描述": "<image>\nDescribe this image in detail.",
        "自定义提示词": custom_prompt if custom_prompt else "<image>\nFree OCR.",
    }

    prompt = prompt_map.get(prompt_type, "<image>\nFree OCR.")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, "input_image.jpg")

    try:
        # 保存上传的图片
        if isinstance(image, str):
            shutil.copy(image, temp_image_path)
        else:
            image.save(temp_image_path)

        # 处理图片
        result_text, result_image = process_single_image(temp_image_path, prompt, temp_dir)

        return result_text, result_image

    except Exception as e:
        return f"处理出错: {str(e)}", None

    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_pdf(pdf_file, prompt_type: str, custom_prompt: str, progress=gr.Progress()):
    """处理上传的 PDF 文件"""
    if pdf_file is None:
        return [], "请上传 PDF 文件"

    if not PDF_SUPPORT:
        return [], "PyMuPDF 未安装，无法处理 PDF 文件。请运行: pip install PyMuPDF"

    # 确定使用的提示词
    prompt_map = {
        "文档转 Markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "图片 OCR": "<image>\n<|grounding|>OCR this image.",
        "纯文字提取（无布局）": "<image>\nFree OCR.",
        "图表解析": "<image>\nParse the figure.",
        "图片详细描述": "<image>\nDescribe this image in detail.",
        "自定义提示词": custom_prompt if custom_prompt else "<image>\nFree OCR.",
    }

    prompt = prompt_map.get(prompt_type, "<image>\nFree OCR.")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 获取 PDF 文件路径
        pdf_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file

        # 将 PDF 转换为图片
        progress(0, desc="正在将 PDF 转换为图片...")
        images = pdf_to_images(pdf_path)
        total_pages = len(images)

        results = []
        all_text = ""

        for i, img in enumerate(images):
            progress((i + 1) / total_pages, desc=f"正在处理第 {i + 1}/{total_pages} 页...")

            # 保存临时图片
            page_dir = os.path.join(temp_dir, f"page_{i + 1}")
            os.makedirs(page_dir, exist_ok=True)
            page_image_path = os.path.join(page_dir, "page.jpg")
            img.save(page_image_path)

            # 处理图片
            try:
                result_text, result_image = process_single_image(page_image_path, prompt, page_dir)

                # 添加页码信息
                page_result = f"\n\n---\n## 第 {i + 1} 页\n\n{result_text}"
                all_text += page_result

                if result_image:
                    results.append((result_image.copy(), f"第 {i + 1} 页"))
                else:
                    results.append((img.copy(), f"第 {i + 1} 页"))

            except Exception as e:
                error_msg = f"第 {i + 1} 页处理失败: {str(e)}"
                all_text += f"\n\n---\n## 第 {i + 1} 页\n\n{error_msg}"
                results.append((img.copy(), f"第 {i + 1} 页 (处理失败)"))

        return results, all_text

    except Exception as e:
        return [], f"处理出错: {str(e)}"

    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# 创建 Gradio 界面

def create_ui():
    # 提示词选项
    prompt_options = [
        "文档转 Markdown",
        "图片 OCR",
        "纯文字提取（无布局）",
        "图表解析",
        "图片详细描述",
        "自定义提示词",
    ]

    # 示例提示词
    example_prompts = """### 示例提示词（保持英文原样使用）:
- **文档转换**: `<image>\\n<|grounding|>Convert the document to markdown.`
- **图片OCR**: `<image>\\n<|grounding|>OCR this image.`
- **纯文字提取**: `<image>\\nFree OCR.`
- **图表解析**: `<image>\\nParse the figure.`
- **图片描述**: `<image>\\nDescribe this image in detail.`
- **定位文字**: `<image>\\nLocate <|ref|>xxxx<|/ref|> in the image.`
"""

    with gr.Blocks(title="DeepSeek-OCR-2 文档解析工具") as demo:
        # 顶部频道信息
        gr.HTML(
            """
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">
                <a href="https://www.youtube.com/@rongyi-ai" target="_blank" style="color: white; text-decoration: none;">
                    🎬 AI 技术分享频道
                </a>
            </h2>
            <p style="color: #f0f0f0; margin: 10px 0 0 0;">
                <a href="https://www.youtube.com/@rongyi-ai" target="_blank" style="color: #f0f0f0;">
                    https://www.youtube.com/@rongyi-ai
                </a>
            </p>
        </div>
        """
        )

        gr.Markdown("# 🔍 DeepSeek-OCR-2 文档解析工具")
        gr.Markdown("基于 DeepSeek-OCR-2 模型，支持图片和 PDF 文件的智能 OCR 解析")

        with gr.Tabs():
            # 图片处理标签页
            with gr.TabItem("📷 图片解析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="上传图片",
                            type="pil",
                            height=400,
                        )

                        image_prompt_type = gr.Dropdown(
                            choices=prompt_options,
                            value="文档转 Markdown",
                            label="选择解析模式",
                        )

                        image_custom_prompt = gr.Textbox(
                            label="自定义提示词",
                            placeholder="输入自定义提示词（选择'自定义提示词'模式时生效）",
                            visible=True,
                        )

                        image_submit_btn = gr.Button("🚀 开始解析", variant="primary")

                    with gr.Column(scale=1):
                        image_result_text = gr.Textbox(
                            label="解析结果",
                            lines=20,
                        )

                        image_result_image = gr.Image(label="布局标注结果")

                gr.Markdown(example_prompts)

                image_submit_btn.click(
                    fn=process_image,
                    inputs=[image_input, image_prompt_type, image_custom_prompt],
                    outputs=[image_result_text, image_result_image],
                )

            # PDF 处理标签页
            with gr.TabItem("📄 PDF 解析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_input = gr.File(
                            label="上传 PDF 文件",
                            file_types=[".pdf"],
                            type="filepath",
                        )

                        pdf_prompt_type = gr.Dropdown(
                            choices=prompt_options,
                            value="文档转 Markdown",
                            label="选择解析模式",
                        )

                        pdf_custom_prompt = gr.Textbox(
                            label="自定义提示词",
                            placeholder="输入自定义提示词（选择'自定义提示词'模式时生效）",
                            visible=True,
                        )

                        pdf_submit_btn = gr.Button("🚀 开始解析 PDF", variant="primary")

                    with gr.Column(scale=2):
                        pdf_preview_image = gr.Image(label="🔍 点击下方缩略图查看大图")

                        pdf_result_gallery = gr.Gallery(
                            label="分页解析结果缩略图（点击查看大图）",
                            columns=6,
                            rows=2,
                            height=200,
                        )

                        pdf_result_text = gr.Textbox(
                            label="完整解析结果",
                            lines=20,
                        )

                gr.Markdown(example_prompts)

                def show_selected_image(evt: gr.SelectData, gallery_data):
                    if gallery_data and evt.index < len(gallery_data):
                        selected = gallery_data[evt.index]
                        if isinstance(selected, tuple):
                            return selected[0]
                        return selected
                    return None

                pdf_result_gallery.select(
                    fn=show_selected_image,
                    inputs=[pdf_result_gallery],
                    outputs=[pdf_preview_image],
                )

                pdf_submit_btn.click(
                    fn=process_pdf,
                    inputs=[pdf_input, pdf_prompt_type, pdf_custom_prompt],
                    outputs=[pdf_result_gallery, pdf_result_text],
                )

        # 底部说明
        gr.Markdown(
            """
        ---
        ### 📋 使用说明

        1. **图片解析**: 支持常见图片格式（JPG、PNG、BMP 等），上传后选择解析模式即可
        2. **PDF 解析**: 上传 PDF 文件后，会自动将每一页转换为图片进行解析，结果分页展示
        3. **解析模式**:
           - **文档转 Markdown**: 适用于文档类图片，会识别布局并转换为 Markdown 格式
           - **图片 OCR**: 通用 OCR 模式，识别图片中的文字
           - **纯文字提取**: 只提取文字，不保留布局信息
           - **图表解析**: 适用于图表、流程图等
           - **图片详细描述**: 对图片内容进行详细描述
           - **自定义提示词**: 使用自定义的提示词进行解析

        ### ⚠️ 注意事项
        - 首次使用时模型加载可能需要一些时间，请耐心等待
        - PDF 文件较大时，处理时间会相应增加
        - 建议使用高清晰度的图片以获得更好的识别效果
        """
        )

    return demo


if __name__ == "__main__":
    print("正在预加载模型...")
    load_model()
    print("模型预加载完成！")

    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
