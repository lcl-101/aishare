import gradio as gr
import asyncio
import sys
import os
import re

# 禁用 vllm v1 引擎
os.environ['VLLM_USE_V1'] = '0'

# 添加本地模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm'))

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
# from config import MODEL_PATH, CROP_MODE  # 不使用 config.py 的 MODEL_PATH
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

# 本地模型配置
MODEL_PATH = 'checkpoints/DeepSeek-OCR'
CROP_MODE = True  # Gundam 模式：动态分辨率

# 注册模型
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# 全局变量存储模型实例
engine = None

async def initialize_model():
    """初始化模型"""
    global engine
    if engine is None:
        print("正在加载 DeepSeek-OCR 模型...")
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("模型加载完成！")
    return engine

def re_match(text):
    """提取文本中的坐标标记"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match)
        else:
            matches_other.append(a_match)
    return matches, matches_image, matches_other

def extract_coordinates_and_label(ref_text, image_width, image_height):
    """从标记中提取坐标和标签"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(f"提取坐标错误: {e}")
        return None
    return (label_type, cor_list)

def draw_bounding_boxes(image, refs):
    """在图片上绘制边界框"""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                # 随机颜色
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20,)
                
                for points in points_list:
                    x1, y1, x2, y2 = points
                    
                    # 转换坐标
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    
                    try:
                        # 绘制边界框
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        
                        # 绘制标签
                        text_x = x1
                        text_y = max(0, y1 - 15)
                        
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                     fill=(255, 255, 255, 200))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception as e:
                        print(f"绘制框错误: {e}")
                        pass
        except Exception as e:
            print(f"处理标记错误: {e}")
            continue
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw

def process_image_with_refs(image, ref_texts):
    """处理图片并添加标记"""
    result_image = draw_bounding_boxes(image, ref_texts)
    return result_image


async def ocr_image_async(image, temperature, max_tokens, ngram_size, window_size, prompt_text):
    """
    对上传的图片进行 OCR 识别（异步版本）
    """
    try:
        # 初始化模型
        llm = await initialize_model()
        
        # 检查图片是否上传
        if image is None:
            return "❌ 请先上传图片", None
        
        # 转换为 RGB 格式
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        # 保存原图用于后续绘制
        original_image = image.copy()
        
        # 使用 DeepseekOCRProcessor 处理图片
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image], 
            bos=True, 
            eos=True, 
            cropping=CROP_MODE
        )
        
        # 设置 logits processors
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=ngram_size, 
                window_size=window_size, 
                whitelist_token_ids={128821, 128822}
            )
        ]
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )
        
        request_id = f"request-{int(time.time())}"
        
        # 准备请求
        request = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image_features}
        }
        
        # 执行 OCR
        print(f"正在执行 OCR 识别... prompt: {prompt_text}")
        full_text = ""
        async for request_output in llm.generate(
            request, sampling_params, request_id
        ):
            if request_output.outputs:
                full_text = request_output.outputs[0].text
        
        print("OCR 识别完成！")
        
        # 如果包含 <|grounding|> 标记，则绘制边界框
        annotated_image = None
        if '<|grounding|>' in prompt_text or '<|ref|>' in full_text:
            try:
                matches_ref, matches_images, matches_other = re_match(full_text)
                if matches_ref:
                    print(f"检测到 {len(matches_ref)} 个标记区域，正在绘制边界框...")
                    annotated_image = process_image_with_refs(original_image, matches_ref)
            except Exception as e:
                print(f"绘制边界框时出错: {e}")
        
        return full_text, annotated_image
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 错误: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None

def ocr_image(image, temperature, max_tokens, ngram_size, window_size, prompt_text):
    """
    对上传的图片进行 OCR 识别（同步包装）
    """
    return asyncio.run(ocr_image_async(image, temperature, max_tokens, ngram_size, window_size, prompt_text))


# 创建 Gradio 界面
with gr.Blocks(title="DeepSeek-OCR WebUI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 DeepSeek-OCR WebUI
        
        基于 DeepSeek-OCR 模型的光学字符识别 (OCR) Web 界面
        """
    )
    
    # 单张图片 OCR
    with gr.Row():
        # 左侧：输入区域
        with gr.Column(scale=1):
            single_image = gr.Image(
                type="pil",
                label="上传图片",
                height=450
            )
            
            prompt_single = gr.Textbox(
                label="提示词 (Prompt)",
                value="<image>\n<|grounding|>Convert the document to markdown.",
                lines=2,
                placeholder="输入提示词，例如: <image>\n<|grounding|>OCR this image."
            )
            
            with gr.Accordion("⚙️ 高级参数", open=False):
                temperature_single = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="Temperature (温度)"
                )
                max_tokens_single = gr.Slider(
                    minimum=512,
                    maximum=16384,
                    value=8192,
                    step=512,
                    label="最大 Token 数"
                )
                ngram_size_single = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=30,
                    step=5,
                    label="N-gram 大小"
                )
                window_size_single = gr.Slider(
                    minimum=30,
                    maximum=150,
                    value=90,
                    step=10,
                    label="窗口大小"
                )
            
            ocr_button_single = gr.Button("🚀 开始识别", variant="primary", size="lg")
        
        # 右侧：输出区域
        with gr.Column(scale=1):
            output_single = gr.Textbox(
                label="识别结果",
                lines=22,
                max_lines=30,
                show_copy_button=True
            )
    
    # 下方：标注图片（可折叠）
    with gr.Row():
        with gr.Column():
            with gr.Accordion("🖼️ 标注图片 (如有)", open=True):
                annotated_image_single = gr.Image(
                    type="pil",
                    label="",
                    height=500,
                    show_label=False
                )
    
    # 添加常用提示词示例
    with gr.Row():
        gr.Examples(
            examples=[
                ["<image>\n<|grounding|>Convert the document to markdown."],
                ["<image>\n<|grounding|>OCR this image."],
                ["<image>\nFree OCR."],
                ["<image>\nParse the figure."],
                ["<image>\nDescribe this image in detail."],
            ],
            inputs=prompt_single,
            label="📝 常用提示词示例"
        )
    
    ocr_button_single.click(
        fn=ocr_image,
        inputs=[
            single_image,
            temperature_single,
            max_tokens_single,
            ngram_size_single,
            window_size_single,
            prompt_single
        ],
        outputs=[output_single, annotated_image_single]
    )
    
    gr.Markdown(
        """
        ---
        ### 📖 使用说明
        
        1. **上传图片** - 支持 PNG、JPG、JPEG、BMP 等格式
        2. **选择提示词** - 可以使用预设的提示词或自定义
        3. **调整参数** - 展开"高级参数"可调整识别参数
        4. **开始识别** - 点击"开始识别"按钮进行 OCR
        5. **查看结果** - 文本结果显示在右侧，标注图片显示在下方
        
        ### 🎯 支持的模式 (Support Modes)
        
        当前开源模型支持以下模式（可在 `config.py` 中配置）:
        
        **原生分辨率 (Native resolution):**
        - **Tiny**: 512×512 (64 vision tokens) ✅
        - **Small**: 640×640 (100 vision tokens) ✅
        - **Base**: 1024×1024 (256 vision tokens) ✅
        - **Large**: 1280×1280 (400 vision tokens) ✅
        
        **动态分辨率 (Dynamic resolution):**
        - **Gundam**: n×640×640 + 1×1024×1024 ✅ (默认配置)
        
        ### 💡 提示词示例 (Prompt Examples)
        
        - `<image>\n<|grounding|>Convert the document to markdown.` - 将文档转换为 Markdown（带位置标注）
        - `<image>\n<|grounding|>OCR this image.` - 带布局的 OCR（其他图片，带位置标注）
        - `<image>\nFree OCR.` - 自由 OCR，不带布局信息
        - `<image>\nParse the figure.` - 解析文档中的图表
        - `<image>\nDescribe this image in detail.` - 详细描述图片内容
        - `<image>\nLocate <|ref|>先天下之忧而忧<|/ref|> in the image.` - 定位指定文本在图片中的位置
        
        ### ⚠️ 注意事项
        
        - 首次运行会自动下载模型，请耐心等待
        - 带 `<|grounding|>` 标记的提示词会包含位置信息，并生成带红框标注的图片
        - 使用 `<|ref|>...<|/ref|>` 可以定位特定文本在图片中的位置
        - 标注图片会用不同颜色的边框标记不同类型的区域（标题用粗框，其他用细框）
        """
    )

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动 DeepSeek-OCR WebUI")
    print("=" * 60)
    
    # 预加载模型
    print("\n⏳ 正在预加载模型，请稍候...")
    asyncio.run(initialize_model())
    print("✅ 模型加载完成！\n")
    
    # 启动 Gradio 应用
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # 端口号
        share=False,             # 是否创建公共链接
        show_error=True          # 显示错误信息
    )
