import gradio as gr
import os
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel
from utils.utils import crop_margin
import cv2
from utils.utils import prepare_image, process_coordinates, parse_layout_string, setup_output_dirs, save_outputs
import tempfile
import re

# HuggingFace model path
MODEL_PATH = "./checkpoints/Dolphin"

doc_examples = [
    "demo/page_imgs/page_1.jpeg",
    "demo/page_imgs/page_2.jpeg",
    "demo/page_imgs/page_3.jpeg",
    "demo/page_imgs/page_4.png",
    "demo/page_imgs/page_5.jpg",
]
element_examples = [
    "demo/element_imgs/para_1.jpg",
    "demo/element_imgs/para_2.jpg",
    "demo/element_imgs/para_3.jpeg",
    "demo/element_imgs/table_1.jpeg",
    "demo/element_imgs/table_2.jpeg",
    "demo/element_imgs/block_formula.jpeg",
    "demo/element_imgs/line_formula.jpeg",
]

# Load model once
class DOLPHIN:
    def __init__(self, model_id_or_path):
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.half()
        self.tokenizer = self.processor.tokenizer

    def chat(self, prompt, image):
        import torch
        is_batch = isinstance(image, list)
        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values.half().to(self.device)
        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt"
        )
        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)
        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
        )
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
            results.append(cleaned)
        if not is_batch:
            return results[0]
        return results

model = DOLPHIN(MODEL_PATH)

def process_elements(layout_results, padded_image, dims, model, max_batch_size=None):
    layout_results = parse_layout_string(layout_results)
    text_elements = []
    table_elements = []
    figure_results = []
    previous_box = None
    reading_order = 0
    for bbox, label in layout_results:
        try:
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0:
                if label == "fig":
                    figure_results.append({
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "text": "",
                        "reading_order": reading_order,
                    })
                else:
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    if label == "tab":
                        table_elements.append(element_info)
                    else:
                        text_elements.append(element_info)
            reading_order += 1
        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue
    recognition_results = figure_results.copy()
    def process_element_batch(elements, model, prompt, max_batch_size=None):
        results = []
        batch_size = len(elements)
        if max_batch_size is not None and max_batch_size > 0:
            batch_size = min(batch_size, max_batch_size)
        for i in range(0, len(elements), batch_size):
            batch_elements = elements[i:i+batch_size]
            crops_list = [elem["crop"] for elem in batch_elements]
            prompts_list = [prompt] * len(crops_list)
            batch_results = model.chat(prompts_list, crops_list)
            for j, result in enumerate(batch_results):
                elem = batch_elements[j]
                results.append({
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                })
        return results
    if text_elements:
        text_results = process_element_batch(text_elements, model, "Read text in the image.", None)
        recognition_results.extend(text_results)
    if table_elements:
        table_results = process_element_batch(table_elements, model, "Parse the table in the image.", None)
        recognition_results.extend(table_results)
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))
    return recognition_results

def recognize_document(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    # 临时保存目录
    with tempfile.TemporaryDirectory() as save_dir:
        pil_image = image
        layout_output = model.chat("Parse the reading order of this document.", pil_image)
        padded_image, dims = prepare_image(pil_image)
        recognition_results = process_elements(layout_output, padded_image, dims, model)
        # 输出 markdown 格式
        from utils.markdown_utils import MarkdownConverter
        markdown_converter = MarkdownConverter()
        markdown_content = markdown_converter.convert(recognition_results)
        # 去除特殊字符，仅保留纯文本
        # 1. 去除 $$...$$ 公式
        text = re.sub(r'\$\$.*?\$\$', '', markdown_content, flags=re.DOTALL)
        # 2. 去除 markdown 标题 ##、# 等
        text = re.sub(r'^#+\\s*', '', text, flags=re.MULTILINE)
        # 3. 去除 URL
        text = re.sub(r'https?://\S+', '', text)
        # 4. 去除多余空行和空格
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()
        return text

def html_table_to_markdown(html):
    import re
    from html import unescape
    # 提取表格内容
    table_match = re.search(r'<table.*?>(.*?)</table>', html, re.DOTALL)
    if not table_match:
        return html
    table_html = table_match.group(1)
    # 提取所有行
    rows = re.findall(r'<tr>(.*?)</tr>', table_html, re.DOTALL)
    table = []
    for row in rows:
        # 提取所有单元格
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
        # 去除 HTML 标签和转义
        cells = [unescape(re.sub(r'<.*?>', '', cell)).replace('\n', ' ').strip() for cell in cells]
        table.append(cells)
    # 生成 markdown
    if not table:
        return html
    md = []
    md.append(' | '.join(table[0]))
    md.append(' | '.join(['---'] * len(table[0])))
    for row in table[1:]:
        md.append(' | '.join(row))
    return '\n'.join(md)

def recognize_element(image, element_type):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    image = crop_margin(image)
    if element_type == "table":
        prompt = "Parse the table in the image."
    elif element_type == "formula":
        prompt = "Read text in the image."
    else:
        prompt = "Read text in the image."
    result = model.chat(prompt, image)
    # 如果是表格，尝试转为markdown
    if element_type == "table" and result.strip().startswith("<table"):
        try:
            return html_table_to_markdown(result)
        except Exception:
            return result
    # 如果是公式，且以 $$ 开头，原样返回
    if element_type == "formula" and result.strip().startswith("$$"):
        return result.strip()
    return result

with gr.Blocks() as demo:
    gr.Markdown("# Dolphin (HuggingFace) 文档与元素识别 WebUI")
    with gr.Tab("文档识别 (Document Recognition)"):
        gr.Markdown("上传文档图片或选择示例，识别文档结构和内容。")
        doc_input = gr.Image(type="filepath", label="上传文档图片 (Upload Document Image)")
        doc_output = gr.Textbox(label="识别结果 (Recognition Result)")
        gr.Examples(
            examples=doc_examples,
            inputs=doc_input,
            outputs=doc_output,
            fn=recognize_document,
            cache_examples=False,
        )
        doc_btn = gr.Button("识别 (Recognize)")
        doc_btn.click(fn=recognize_document, inputs=doc_input, outputs=doc_output)
    with gr.Tab("元素识别 (Element Recognition)"):
        gr.Markdown("上传元素图片或选择示例，识别文本、表格或公式内容。")
        elem_input = gr.Image(type="filepath", label="上传元素图片 (Upload Element Image)")
        elem_type = gr.Radio(["text", "table", "formula"], value="text", label="元素类型 (Element Type)")
        elem_output = gr.Textbox(label="识别结果 (Recognition Result)")
        gr.Examples(
            examples=[[ex, "text"] for ex in element_examples],
            inputs=[elem_input, elem_type],
            outputs=elem_output,
            fn=recognize_element,
            cache_examples=False,
        )
        elem_btn = gr.Button("识别 (Recognize)")
        elem_btn.click(fn=recognize_element, inputs=[elem_input, elem_type], outputs=elem_output)

demo.launch(server_name="0.0.0.0")
