"""
Dolphin Document Parser - Gradio Web Application
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import os
import json
import tempfile
import shutil
from PIL import Image
import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from utils.utils import (
    resize_img, 
    parse_layout_string, 
    process_coordinates, 
    setup_output_dirs, 
    save_outputs,
    convert_pdf_to_images,
    save_figure_to_local,
    visualize_layout,
)
from utils.markdown_utils import MarkdownConverter


# Global model instance
model = None


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model"""
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id_or_path)
        self.model.eval()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if self.device == "cuda":
            self.model = self.model.bfloat16()
        else:
            self.model = self.model.float()
        
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

    def chat(self, prompt, image):
        is_batch = isinstance(image, list)
        
        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        assert len(images) == len(prompts)
        
        processed_images = [resize_img(img) for img in images]
        all_messages = []
        for img, question in zip(processed_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": question}
                    ],
                }
            ]
            all_messages.append(messages)

        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in all_messages
        ]

        all_image_inputs = []
        for msgs in all_messages:
            image_inputs, _ = process_vision_info(msgs)
            all_image_inputs.extend(image_inputs)

        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            temperature=None,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        results = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        if not is_batch:
            return results[0]
        return results


def load_model(model_path="./checkpoints/Dolphin-v2"):
    """Load the model globally"""
    global model
    if model is None:
        print(f"Loading model from {model_path}...")
        model = DOLPHIN(model_path)
        print("Model loaded successfully!")
    return model


def check_bbox_overlap(layout_results_list, image, overlap_threshold=0.5):
    """Check if there's significant overlap between bounding boxes"""
    if len(layout_results_list) < 2:
        return False
    
    overlap_count = 0
    total_pairs = 0
    
    for i in range(len(layout_results_list)):
        for j in range(i + 1, len(layout_results_list)):
            bbox1 = layout_results_list[i][0]
            bbox2 = layout_results_list[j][0]
            
            x1_1, y1_1, x2_1, y2_1 = process_coordinates(bbox1, image)
            x1_2, y1_2, x2_2, y2_2 = process_coordinates(bbox2, image)
            
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            if xi2 > xi1 and yi2 > yi1:
                inter_area = (xi2 - xi1) * (yi2 - yi1)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                min_area = min(area1, area2)
                
                if min_area > 0 and inter_area / min_area > overlap_threshold:
                    overlap_count += 1
            
            total_pairs += 1
    
    return total_pairs > 0 and overlap_count / total_pairs > 0.3


def process_elements_for_page(layout_results, image, model_instance, max_batch_size=4, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding"""
    layout_results_list = parse_layout_string(layout_results)
    if not layout_results_list or not (layout_results.startswith("[") and layout_results.endswith("]")):
        layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]
    elif len(layout_results_list) > 1 and check_bbox_overlap(layout_results_list, image):
        layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]
        
    tab_elements = []      
    equ_elements = []     
    code_elements = []    
    text_elements = []     
    figure_results = []    
    reading_order = 0

    for bbox, label, tags in layout_results_list:
        try:
            if label == "distorted_page":
                x1, y1, x2, y2 = 0, 0, *image.size
                pil_crop = image
            else:
                x1, y1, x2, y2 = process_coordinates(bbox, image)
                pil_crop = image.crop((x1, y1, x2, y2))

            if pil_crop.size[0] > 3 and pil_crop.size[1] > 3:
                if label == "fig":
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    figure_results.append({
                        "label": label,
                        "text": f"![Figure](figures/{figure_filename})",
                        "figure_path": f"figures/{figure_filename}",
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    })
                else:
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    }
                    
                    if label == "tab":
                        tab_elements.append(element_info)
                    elif label == "equ":
                        equ_elements.append(element_info)
                    elif label == "code":
                        code_elements.append(element_info)
                    else:
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    recognition_results = figure_results.copy()
    
    def process_element_batch(elements, prompt, max_batch_size):
        results = []
        batch_size = len(elements) if max_batch_size is None else min(len(elements), max_batch_size)
        
        for i in range(0, len(elements), batch_size):
            batch_elements = elements[i:i+batch_size]
            crops_list = [elem["crop"] for elem in batch_elements]
            prompts_list = [prompt] * len(crops_list)
            
            batch_results = model_instance.chat(prompts_list, crops_list)
            
            for j, result in enumerate(batch_results):
                elem = batch_elements[j]
                results.append({
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                    "tags": elem["tags"],
                })
        return results
    
    if tab_elements:
        results = process_element_batch(tab_elements, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if equ_elements:
        results = process_element_batch(equ_elements, "Read formula in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if code_elements:
        results = process_element_batch(code_elements, "Read code in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if text_elements:
        results = process_element_batch(text_elements, "Read text in the image.", max_batch_size)
        recognition_results.extend(results)

    recognition_results.sort(key=lambda x: x.get("reading_order", 0))
    return recognition_results


# ==================== Page-level Parsing ====================
def page_level_parsing(image, max_batch_size=4, progress=gr.Progress()):
    """Process page-level document parsing"""
    if image is None:
        return None, "请上传图片", "{}"
    
    progress(0, desc="加载模型...")
    model_instance = load_model()
    
    progress(0.2, desc="解析页面布局...")
    
    # Convert to PIL Image if needed
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("RGB")
    
    # Create temp directory for outputs
    temp_dir = tempfile.mkdtemp()
    setup_output_dirs(temp_dir)
    image_name = "uploaded_image"
    
    # Stage 1: Layout parsing
    layout_output = model_instance.chat("Parse the reading order of this document.", pil_image)
    
    progress(0.5, desc="解析文档元素...")
    
    # Stage 2: Element-level parsing
    recognition_results = process_elements_for_page(
        layout_output, pil_image, model_instance, max_batch_size, temp_dir, image_name
    )
    
    progress(0.8, desc="生成输出...")
    
    # Generate markdown
    markdown_converter = MarkdownConverter()
    markdown_content = markdown_converter.convert(recognition_results)
    
    # Generate visualization
    vis_path = os.path.join(temp_dir, "layout_visualization", f"{image_name}_layout.png")
    visualize_layout(pil_image, recognition_results, vis_path)
    vis_image = Image.open(vis_path) if os.path.exists(vis_path) else None
    
    # JSON output
    json_output = json.dumps(recognition_results, ensure_ascii=False, indent=2)
    
    progress(1.0, desc="完成!")
    
    return vis_image, markdown_content, json_output


# ==================== Element-level Parsing ====================
def element_level_parsing(image, element_type, progress=gr.Progress()):
    """Process element-level parsing"""
    if image is None:
        return "请上传图片", "{}"
    
    progress(0, desc="加载模型...")
    model_instance = load_model()
    
    progress(0.3, desc=f"解析 {element_type}...")
    
    # Convert to PIL Image
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("RGB")
    
    # Select prompt based on element type
    prompts = {
        "table": ("Parse the table in the image.", "tab"),
        "formula": ("Read formula in the image.", "equ"),
        "text": ("Read text in the image.", "para"),
        "code": ("Read code in the image.", "code"),
    }
    
    prompt, label = prompts.get(element_type, ("Read text in the image.", "para"))
    
    progress(0.6, desc="模型推理中...")
    result = model_instance.chat(prompt, pil_image)
    
    recognition_results = [{
        "label": label,
        "text": result.strip(),
    }]
    
    json_output = json.dumps(recognition_results, ensure_ascii=False, indent=2)
    
    progress(1.0, desc="完成!")
    
    return result.strip(), json_output


# ==================== Layout Parsing ====================
def layout_parsing(image, progress=gr.Progress()):
    """Process layout detection"""
    if image is None:
        return None, "请上传图片", "{}"
    
    progress(0, desc="加载模型...")
    model_instance = load_model()
    
    progress(0.3, desc="解析布局...")
    
    # Convert to PIL Image
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("RGB")
    
    # Parse layout
    layout_results = model_instance.chat("Parse the reading order of this document.", pil_image)
    
    progress(0.6, desc="处理结果...")
    
    # Parse the layout string
    layout_results_list = parse_layout_string(layout_results)
    if not layout_results_list or not (layout_results.startswith("[") and layout_results.endswith("]")):
        layout_results_list = [([0, 0, *pil_image.size], 'distorted_page', [])]
    
    # Map bbox to original image coordinates
    recognition_results = []
    reading_order = 0
    for bbox, label, tags in layout_results_list:
        x1, y1, x2, y2 = process_coordinates(bbox, pil_image)
        recognition_results.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "text": "",
            "reading_order": reading_order,
            "tags": tags,
        })
        reading_order += 1
    
    progress(0.8, desc="生成可视化...")
    
    # Generate visualization
    temp_dir = tempfile.mkdtemp()
    vis_path = os.path.join(temp_dir, "layout_vis.png")
    visualize_layout(pil_image, recognition_results, vis_path)
    vis_image = Image.open(vis_path) if os.path.exists(vis_path) else None
    
    # Create readable layout description
    layout_desc = "## 布局检测结果\n\n"
    for result in recognition_results:
        layout_desc += f"**{result['reading_order']}**: {result['label']}"
        if result['tags']:
            layout_desc += f" | 标签: {', '.join(result['tags'])}"
        layout_desc += f"\n   - 位置: {result['bbox']}\n\n"
    
    json_output = json.dumps(recognition_results, ensure_ascii=False, indent=2)
    
    progress(1.0, desc="完成!")
    
    return vis_image, layout_desc, json_output


# ==================== Gradio Interface ====================
def create_demo():
    """Create Gradio interface with tabs for different functionalities"""
    
    # Get example images
    page_examples = [
        os.path.join("demo/page_imgs", f) 
        for f in ["page_1.png", "page_2.jpeg", "page_3.jpeg", "page_4.png", "page_5.jpg"]
        if os.path.exists(os.path.join("demo/page_imgs", f))
    ]
    
    element_examples = {
        "table": ["demo/element_imgs/table.jpg"],
        "formula": ["demo/element_imgs/block_formula.jpeg", "demo/element_imgs/line_formula.jpeg"],
        "text": ["demo/element_imgs/para_1.jpg", "demo/element_imgs/para_2.jpg", "demo/element_imgs/para_3.jpeg"],
        "code": ["demo/element_imgs/code.jpeg"],
    }
    
    with gr.Blocks(
        title="🐬 Dolphin 文档解析器",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .tab-nav button { font-size: 16px !important; }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🐬 Dolphin 文档解析器
            
            **Dolphin** 是一个多功能的文档解析模型，支持页面级解析、元素级解析和布局检测。
            
            - **📄 页面解析**: 完整解析文档页面，提取所有元素并生成 Markdown
            - **🧩 元素解析**: 单独解析表格、公式、文本或代码
            - **🎨 布局检测**: 检测文档布局结构和阅读顺序
            """
        )
        
        with gr.Tabs():
            # =============== Tab 1: Page-level Parsing ===============
            with gr.TabItem("📄 页面解析 (Page Parsing)"):
                gr.Markdown("### 上传文档图片，获取完整的文档解析结果")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        page_input = gr.Image(
                            label="上传文档图片",
                            type="numpy",
                            height=400
                        )
                        page_batch_size = gr.Slider(
                            minimum=1, 
                            maximum=16, 
                            value=4, 
                            step=1,
                            label="批处理大小 (Batch Size)"
                        )
                        page_btn = gr.Button("🚀 开始解析", variant="primary", size="lg")
                        
                        gr.Markdown("### 📚 示例图片")
                        gr.Examples(
                            examples=page_examples,
                            inputs=page_input,
                            label="点击选择示例"
                        )
                    
                    with gr.Column(scale=1):
                        page_vis_output = gr.Image(
                            label="布局可视化",
                            height=400
                        )
                        
                        with gr.Tabs():
                            with gr.TabItem("📝 Markdown"):
                                page_md_output = gr.Markdown(
                                    label="Markdown 输出",
                                    value=""
                                )
                            with gr.TabItem("📊 JSON"):
                                page_json_output = gr.Code(
                                    label="JSON 输出",
                                    language="json"
                                )
                
                page_btn.click(
                    fn=page_level_parsing,
                    inputs=[page_input, page_batch_size],
                    outputs=[page_vis_output, page_md_output, page_json_output]
                )
            
            # =============== Tab 2: Element-level Parsing ===============
            with gr.TabItem("🧩 元素解析 (Element Parsing)"):
                gr.Markdown("### 解析单个文档元素：表格、公式、文本或代码")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        element_input = gr.Image(
                            label="上传元素图片",
                            type="numpy",
                            height=350
                        )
                        element_type = gr.Radio(
                            choices=["table", "formula", "text", "code"],
                            value="text",
                            label="元素类型",
                            info="选择要解析的元素类型"
                        )
                        element_btn = gr.Button("🚀 开始解析", variant="primary", size="lg")
                        
                        gr.Markdown("### 📚 示例图片")
                        
                        # 构建带元素类型的示例列表
                        table_examples = [[img, "table"] for img in element_examples.get("table", [])]
                        formula_examples = [[img, "formula"] for img in element_examples.get("formula", [])]
                        text_examples = [[img, "text"] for img in element_examples.get("text", [])]
                        code_examples = [[img, "code"] for img in element_examples.get("code", [])]
                        
                        with gr.Accordion("表格示例", open=False):
                            gr.Examples(
                                examples=table_examples,
                                inputs=[element_input, element_type],
                                label="表格"
                            )
                        
                        with gr.Accordion("公式示例", open=False):
                            gr.Examples(
                                examples=formula_examples,
                                inputs=[element_input, element_type],
                                label="公式"
                            )
                        
                        with gr.Accordion("文本示例", open=False):
                            gr.Examples(
                                examples=text_examples,
                                inputs=[element_input, element_type],
                                label="文本"
                            )
                        
                        with gr.Accordion("代码示例", open=False):
                            gr.Examples(
                                examples=code_examples,
                                inputs=[element_input, element_type],
                                label="代码"
                            )
                    
                    with gr.Column(scale=1):
                        element_text_output = gr.Textbox(
                            label="解析结果",
                            lines=15,
                            max_lines=30
                        )
                        element_json_output = gr.Code(
                            label="JSON 输出",
                            language="json"
                        )
                
                element_btn.click(
                    fn=element_level_parsing,
                    inputs=[element_input, element_type],
                    outputs=[element_text_output, element_json_output]
                )
            
            # =============== Tab 3: Layout Parsing ===============
            with gr.TabItem("🎨 布局检测 (Layout Detection)"):
                gr.Markdown("### 检测文档布局结构和阅读顺序")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        layout_input = gr.Image(
                            label="上传文档图片",
                            type="numpy",
                            height=400
                        )
                        layout_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")
                        
                        gr.Markdown("### 📚 示例图片")
                        gr.Examples(
                            examples=page_examples,
                            inputs=layout_input,
                            label="点击选择示例"
                        )
                    
                    with gr.Column(scale=1):
                        layout_vis_output = gr.Image(
                            label="布局可视化",
                            height=400
                        )
                        
                        with gr.Tabs():
                            with gr.TabItem("📋 布局描述"):
                                layout_desc_output = gr.Markdown(
                                    label="布局描述"
                                )
                            with gr.TabItem("📊 JSON"):
                                layout_json_output = gr.Code(
                                    label="JSON 输出",
                                    language="json"
                                )
                
                layout_btn.click(
                    fn=layout_parsing,
                    inputs=[layout_input],
                    outputs=[layout_vis_output, layout_desc_output, layout_json_output]
                )
        
        gr.Markdown(
            """
            ---
            ### 使用说明
            
            1. **页面解析**: 上传完整的文档页面图片，系统会自动检测布局并解析所有元素
            2. **元素解析**: 上传裁剪好的单个元素图片（表格/公式/文本/代码），进行针对性解析
            3. **布局检测**: 上传文档图片，仅检测布局结构，不进行内容识别
            
            ### 支持的元素类型
            - `table`: 表格
            - `formula`: 数学公式
            - `text`: 普通文本段落
            - `code`: 代码块
            
            ---
            **Powered by Dolphin** | [GitHub](https://github.com/bytedance/Dolphin)
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
