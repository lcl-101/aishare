import gradio as gr
import torch
import os
from PIL import Image
import cairosvg
import io
import tempfile
import argparse
import gc
import yaml
import glob


from decoder import SketchDecoder
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tokenizer import SVGTokenizer

with open('checkpoints/OmniSVG/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = None
processor = None
sketch_decoder = None
svg_tokenizer = None

# 系统提示词
SYSTEM_PROMPT = "You are a multimodal SVG generation assistant capable of generating SVG code from both text descriptions and images."
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SVG生成器服务')
    parser.add_argument('--listen', type=str, default='0.0.0.0', 
                       help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=7860, 
                       help='端口号 (默认: 7860)')
    parser.add_argument('--share', action='store_true', 
                       help='启用gradio分享链接')
    parser.add_argument('--debug', action='store_true', 
                       help='启用调试模式')
    return parser.parse_args()

def load_models():
    """加载模型"""
    global tokenizer, processor, sketch_decoder, svg_tokenizer
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("checkpoints/Qwen2.5-VL-3B-Instruct", padding_side="left")
        processor = AutoProcessor.from_pretrained("checkpoints/Qwen2.5-VL-3B-Instruct", padding_side="left")

        sketch_decoder = SketchDecoder()
        
        sketch_weight_path = "checkpoints/OmniSVG/pytorch_model.bin"
        sketch_decoder.load_state_dict(torch.load(sketch_weight_path))
        sketch_decoder = sketch_decoder.to(device).eval()

        svg_tokenizer = SVGTokenizer('checkpoints/OmniSVG/config.yaml')


def process_and_resize_image(image_input, target_size=(200, 200)):
    """处理并调整图像到目标尺寸"""
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.fromarray(image_input)
    
    
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image

def get_example_images():
    """从示例目录获取示例图片"""
    example_dir = "./examples"
    example_images = []
    
    if os.path.exists(example_dir):
        for ext in SUPPORTED_FORMATS:
            pattern = os.path.join(example_dir, f"*{ext}")
            example_images.extend(glob.glob(pattern))
        
        example_images.sort()
    
    return example_images

def process_text_to_svg(text_description):
    """处理文本转SVG任务"""
    load_models()
    
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Task: text-to-svg\nDescription: {text_description}\nGenerate SVG code based on the above description."}
        ]
    }]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input], 
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    pixel_values = None
    image_grid_thw = None
    
    return input_ids, attention_mask, pixel_values, image_grid_thw

def process_image_to_svg(image_path):
    """处理图像转SVG任务"""
    load_models()
    
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user", 
        "content": [
            {"type": "text", "text": f"Task: image-to-svg\nGenerate SVG code that accurately represents the following image."},
            {"type": "image", "image": image_path},
        ]
    }]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text_input], 
        images=image_inputs,
        truncation=True, 
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    pixel_values = inputs['pixel_values'].to(device) if 'pixel_values' in inputs else None
    image_grid_thw = inputs['image_grid_thw'].to(device) if 'image_grid_thw' in inputs else None
    
    return input_ids, attention_mask, pixel_values, image_grid_thw

def generate_svg(input_ids, attention_mask, pixel_values=None, image_grid_thw=None, task_type="image-to-svg"):
    """生成SVG"""
    try:
        # 在生成前清理内存
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"正在生成{task_type}的SVG...")
        
        # 生成配置，可调整以获得更好的结果
        if task_type == "image-to-svg":
            #图像转SVG配置
            gen_config = dict(
                do_sample=True,
                temperature=0.1,
                top_p=0.001,
                top_k=1,
                num_beams=5,
                repetition_penalty=1.05,
            )
        else:
            #文本转SVG配置
            gen_config = dict(
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                early_stopping=True,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 生成SVG
        model_config = config['model']
        max_length = model_config['max_length']
        output_ids = torch.ones(1, max_length).long().to(device) * model_config['eos_token_id']
        
        with torch.no_grad():
            results = sketch_decoder.transformer.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_length-1,
                num_return_sequences=1,
                bos_token_id=model_config['bos_token_id'],
                eos_token_id=model_config['eos_token_id'],
                pad_token_id=model_config['pad_token_id'],
                use_cache=True,
                **gen_config
            )
            results = results[:, :max_length-1]
            output_ids[:, :results.shape[1]] = results
        
            generated_xy, generated_colors = svg_tokenizer.process_generated_tokens(output_ids)
            print(f"生成的XY坐标数量: {len(generated_xy) if generated_xy is not None else 0}")
            print(f"生成的颜色数量: {len(generated_colors) if generated_colors is not None else 0}")

        svg_tensors = svg_tokenizer.raster_svg(generated_xy)
        print(f"SVG张量数量: {len(svg_tensors) if svg_tensors is not None else 0}")
        if svg_tensors is None or len(svg_tensors) == 0 or svg_tensors[0] is None:
            return "错误: 未生成有效的SVG路径", None
            
        print('正在创建SVG...')

        svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], generated_colors)
        
        svg_str = svg.to_str()
        print(f"生成的SVG长度: {len(svg_str)}")
        
        # 转换为PNG用于可视化
        try:
            # 尝试使用cairosvg转换
            png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
            png_image = Image.open(io.BytesIO(png_data))
            print(f"PNG图像尺寸: {png_image.size}")
            
            return svg_str, png_image
        except Exception as png_error:
            print(f"PNG转换错误: {png_error}")
            # 如果PNG转换失败，只返回SVG代码
            return svg_str, None
                
    except Exception as e:
        print(f"生成错误: {e}")
        import traceback
        traceback.print_exc()
        return f"错误: {e}", None

def gradio_image_to_svg(image):
    """Gradio界面函数 - 图像转SVG"""
    if image is None:
        return "请上传一张图片", None
    processed_image = process_and_resize_image(image)
    
    # 保存临时图像文件
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        processed_image.save(tmp_file.name, format='PNG')
        tmp_path = tmp_file.name
    
    try:
        input_ids, attention_mask, pixel_values, image_grid_thw = process_image_to_svg(tmp_path)
        svg_code, png_image = generate_svg(input_ids, attention_mask, pixel_values, image_grid_thw, "image-to-svg")
        return svg_code, png_image
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def gradio_text_to_svg(text_description):
    """Gradio界面函数 - 文本转SVG"""
    if not text_description or text_description.strip() == "":
        return "请输入描述", None
    
    input_ids, attention_mask, pixel_values, image_grid_thw = process_text_to_svg(text_description)
    svg_code, png_image = generate_svg(input_ids, attention_mask, pixel_values, image_grid_thw, "text-to-svg")
    return svg_code, png_image

def create_interface():
    # 示例文本 - 显示用的中文描述
    example_texts_display = [
        "一个圆角的红色心形",
        "一个五角黄色星星", 
        "带有向上箭头的云朵图标，象征上传或云存储",
        "一个棕色巧克力条，分为四个方形段，具有光泽质感",
        "一个彩色搬家卡车图标，带有红色和橙色货柜",
        "一个灰色挂锁图标，象征安全和保护",
        "一个浅蓝色T恤图标，用粗蓝色边框勾勒",
        "一个穿蓝色衬衫和深色裤子的人，一只手插在口袋里，另一只手向外做手势",
    ]
    
    # 实际填入的英文描述
    example_texts_english = [
        "A red heart shape with rounded corners.",
        "A yellow star with five points.",
        "Cloud icon with an upward arrow symbolizes uploading or cloud storage.",
        "A brown chocolate bar is depicted in four square segments with a shiny glossy finish.",
        "A colorful moving truck icon with a red and orange cargo container.",
        "A gray padlock icon symbolizes security and protection.",
        "A light blue T-shirt icon is outlined with a bold blue border.",
        "A person in a blue shirt and dark pants stands with one hand in a pocket gesturing outward.",
    ]
    example_images = get_example_images()
    
    with gr.Blocks(title="OmniSVG 演示页面", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# OmniSVG 演示页面")
        gr.Markdown("从图像或文本描述生成SVG代码")
        
        with gr.Tabs():
            # 图像转SVG标签页
            with gr.TabItem("图像转SVG"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="输入图像", 
                            type="pil",
                            image_mode="RGBA"
                        )
                        if example_images:
                            gr.Examples(
                                examples=example_images,
                                inputs=[image_input],
                                label="示例图像 (点击使用)",
                                examples_per_page=10
                            )
                        image_generate_btn = gr.Button("生成SVG", variant="primary")
                    
                    with gr.Column():
                        image_svg_output = gr.Textbox(
                            label="生成的SVG代码", 
                            lines=10,
                            max_lines=20,
                            show_copy_button=True
                        )
                        image_png_preview = gr.Image(label="SVG预览", type="pil")
                
                image_generate_btn.click(
                    fn=gradio_image_to_svg,
                    inputs=[image_input],
                    outputs=[image_svg_output, image_png_preview],
                    queue=True
                )
            
            # 文本转SVG标签页
            with gr.TabItem("文本转SVG"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="描述",
                            placeholder="输入SVG描述，例如：一个红色圆圈内有一个蓝色方块",
                            lines=3
                        )
                        
                        # 添加示例文本
                        def create_examples():
                            examples = []
                            for i, (display_text, english_text) in enumerate(zip(example_texts_display, example_texts_english)):
                                examples.append([english_text])  # 实际填入的是英文
                            return examples
                        
                        # 创建自定义的示例组件
                        with gr.Row():
                            gr.Markdown("**示例描述 (点击使用):**")
                        
                        # 创建示例按钮
                        example_buttons = []
                        for i, (display_text, english_text) in enumerate(zip(example_texts_display, example_texts_english)):
                            btn = gr.Button(display_text, size="sm", variant="secondary")
                            btn.click(lambda eng=english_text: eng, outputs=text_input)
                            example_buttons.append(btn)
                        
                        text_generate_btn = gr.Button("生成SVG", variant="primary")
                    
                    with gr.Column():
                        text_svg_output = gr.Textbox(
                            label="生成的SVG代码", 
                            lines=10,
                            max_lines=20,
                            show_copy_button=True
                        )
                        text_png_preview = gr.Image(label="SVG预览", type="pil")
                
                text_generate_btn.click(
                    fn=gradio_text_to_svg,
                    inputs=[text_input],
                    outputs=[text_svg_output, text_png_preview],
                    queue=True
                )
        
        # 添加使用说明
        gr.Markdown("""
        ## 使用说明
        - **图像转SVG**: 上传一张PNG图片并点击"生成SVG"
        - **文本转SVG**: 输入文本描述或点击示例，然后点击"生成SVG"
        
        """)
    
    return demo

if __name__ == "__main__":
    # 设置环境变量以避免tokenizer并行化警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = parse_args()
    
    # 在启动前加载模型
    print("正在加载模型...")
    load_models()
    print("模型加载成功!")
    
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        server_name=args.listen,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )
