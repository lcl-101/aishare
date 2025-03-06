import torch
import base64
import urllib.request
import json
import re
import PyPDF2  # 用于获取 PDF 页数

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# 初始化模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "checkpoints/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained("checkpoints/Qwen2-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 下载示例 PDF 文件
#pdf_url = "https://molmo.allenai.org/paper.pdf"
pdf_path = "./163.pdf"
#urllib.request.urlretrieve(pdf_url, pdf_path)

# 自动获取 PDF 的总页数
with open(pdf_path, "rb") as f:
    pdf_reader = PyPDF2.PdfReader(f)
    total_pages = len(pdf_reader.pages)
print(f"PDF 总页数: {total_pages}")

# 遍历每一页
for page in range(1, total_pages + 1):
    print(f"正在处理第 {page} 页...")

    # 将当前页渲染成图片，并编码成 Base64 格式
    image_base64 = render_pdf_to_base64png(pdf_path, page, target_longest_image_dim=1024)
    
    # 提取当前页的锚文本，并构建微调提示
    anchor_text = get_anchor_text(pdf_path, page, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text)
    
    # 构建当前页的完整提示消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    
    # 应用聊天模板和处理器
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Print the prompt for debugging, showing only the text part
    print("Prompt text part:", prompt)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
    
    # 准备输入
    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 为当前页生成模型输出
    output = model.generate(
        **inputs,
        temperature=0.8,
        max_new_tokens=50000,
        num_return_sequences=1,
        do_sample=True,
    )
    
    # 解码生成的输出
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    # 调试：打印原始生成结果以检查格式，并提取 "natural_text"
    for out in text_output:
        print(f"第 {page} 页的原始模型输出:")
        print(out)
        try:
            parsed = json.loads(out)
            natural_text = parsed.get("natural_text", "未找到 natural_text 键")
        except Exception as e:
            # 如果 JSON 解析失败，尝试使用常规的正则表达式匹配（支持换行）
            match = re.search(r'"natural_text"\s*:\s*"([\s\S]+?)"(,|}|$)', out)
            if match:
                natural_text = match.group(1)
            else:
                # 如果常规匹配失败，尝试宽松匹配：从 "natural_text" 开始，到字符串末尾
                match = re.search(r'"natural_text"\s*:\s*"([\s\S]+)', out)
                if match:
                    natural_text = match.group(1)
                    # 去除末尾多余的字符
                    natural_text = natural_text.strip().rstrip('",}')
                else:
                    natural_text = "提取 natural_text 失败"
        print(f"第 {page} 页的输出:")
        print(natural_text)
    
    print("=" * 50)