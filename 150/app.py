import os
import multiprocessing
import requests
import io
import gradio as gr
from PIL import Image
import fitz  # PyMUPDF
import decord
import numpy as np

# 设置环境变量 - 在导入任何CUDA相关库之前
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_USE_MULTIPROCESS"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_USE_V1"] = "0"
os.environ["MAX_JOBS"] = "1"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# 在最开始就设置多进程方法
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# 全局变量
model_path = "checkpoints/Kimi-VL-A3B-Thinking-2506"
llm = None
processor = None
sampling_params = None

def initialize_model():
    """初始化模型"""
    global llm, processor, sampling_params
    if llm is None:
        print("Initializing model...")
        llm = LLM(
            model_path,
            trust_remote_code=True,
            max_num_seqs=1,
            max_model_len=131072,
            limit_mm_per_prompt={"image": 16},
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            dtype="bfloat16",
            disable_log_stats=True,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        sampling_params = SamplingParams(max_tokens=32768, temperature=0.8)
        print("Model initialized successfully!")

def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷"):
    """提取思考过程和总结"""
    if bot in text and eot not in text:
        return "", ""
    if eot in text:
        thinking = text[text.index(bot) + len(bot):text.index(eot)].strip()
        summary = text[text.index(eot) + len(eot):].strip()
        return thinking, summary
    return "", text

def download_and_cache_file(url, filename):
    """下载文件到examples目录"""
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    file_path = os.path.join(examples_dir, filename)
    
    if os.path.exists(file_path):
        print(f"File {filename} already exists in examples directory")
        return file_path
    
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded {filename} to examples directory")
    return file_path

def process_single_image(image, question):
    """处理单张图像"""
    if image is None:
        return "请上传一张图片", ""
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": ""},
            {"type": "text", "text": question}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = llm.generate([{"prompt": text, "multi_modal_data": {"image": image}}], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    thinking, summary = extract_thinking_and_summary(generated_text)
    
    result = f"**思考过程:**\n{thinking}\n\n**总结:**\n{summary}" if thinking else summary
    return result, generated_text

def process_pdf(pdf_file, question):
    """处理PDF文件"""
    if pdf_file is None:
        return "请上传一个PDF文件", ""
    
    # 转换PDF为图像
    doc = fitz.open(pdf_file.name)
    all_images = []
    
    max_pages = min(len(doc), 8)  # 限制最多8页
    
    for page_num in range(max_pages):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        all_images.append(image)
    
    doc.close()
    
    messages = [
        {"role": "user", "content": [
            *[{"type": "image", "image": ""} for _ in all_images],
            {"type": "text", "text": question}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = llm.generate([{"prompt": text, "multi_modal_data": {"image": all_images}}], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    thinking, summary = extract_thinking_and_summary(generated_text)
    
    result = f"**处理了{len(all_images)}页PDF**\n\n**思考过程:**\n{thinking}\n\n**总结:**\n{summary}" if thinking else f"**处理了{len(all_images)}页PDF**\n\n{summary}"
    return result, generated_text

def resize_image_and_convert_to_pil(image, max_size=448):
    """调整图像尺寸"""
    with Image.fromarray(image) as img:
        width, height = img.size
        max_side = max(width, height)
        scale_ratio = max_size / max_side
        new_width, new_height = int(width * scale_ratio), int(height * scale_ratio)
        return img.resize((new_width, new_height))

def fmt_timestamp(timestamp_sec):
    """格式化时间戳"""
    hours = int(timestamp_sec // 3600)
    minutes = int((timestamp_sec % 3600) // 60)
    seconds = int(timestamp_sec % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_video(video_file, question):
    """处理视频文件"""
    if video_file is None:
        return "请上传一个视频文件", ""
    
    try:
        # 提取视频帧
        vr = decord.VideoReader(video_file.name)
        fps = vr.get_avg_fps()
        video_duration = int(len(vr) / fps)
        sample_fps = 0.5  # 每2秒采样一帧
        sample_frames = min(int(video_duration * sample_fps) + 1, 12)  # 最多12帧
        
        frame_inds = np.linspace(0, len(vr) - 1, sample_frames).round().astype(int)
        frames = vr.get_batch(frame_inds).asnumpy()
        timestamps = (frame_inds / fps).astype(np.int32)
        
        # 转换帧为PIL图像
        images = [resize_image_and_convert_to_pil(frame) for frame in frames]
        
        # 构建消息
        contents = []
        for timestamp in timestamps:
            contents.append({"type": "text", "text": fmt_timestamp(timestamp)})
            contents.append({"type": "image", "image": ""})
        contents.append({"type": "text", "text": question})
        
        messages = [{"role": "user", "content": contents}]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = llm.generate([{"prompt": text, "multi_modal_data": {"image": images}}], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        thinking, summary = extract_thinking_and_summary(generated_text)
        
        result = f"**处理了{len(images)}帧视频**\n\n**思考过程:**\n{thinking}\n\n**总结:**\n{summary}" if thinking else f"**处理了{len(images)}帧视频**\n\n{summary}"
        return result, generated_text
        
    except Exception as e:
        return f"处理视频时出错: {str(e)}", ""

def process_gui_agent(image, task):
    """处理GUI Agent任务"""
    if image is None:
        return "请上传一张屏幕截图", ""
    
    system_prompt = """你是一个GUI智能助手。你会收到一个任务和计算机屏幕的截图。你需要执行操作并生成pyautogui代码来完成任务。请按以下格式提供回复：

## 操作步骤:
提供清晰、简洁、可执行的指令。

## 代码:
生成相应的Python代码片段，使用pyautogui库，通过标准化屏幕坐标（0到1之间的值）点击识别的UI元素。脚本应该通过将标准化坐标转换为实际像素位置来动态适应当前屏幕分辨率。"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": ""},
            {"type": "text", "text": f"## Task Instruction:\n{task}"}
        ]}
    ]
    
    # 使用较低的温度以获得更精确的结果
    gui_sampling_params = SamplingParams(max_tokens=8192, temperature=0.2)
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = llm.generate([{"prompt": text, "multi_modal_data": {"image": image}}], gui_sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    thinking, summary = extract_thinking_and_summary(generated_text)
    
    result = f"**思考过程:**\n{thinking}\n\n**操作指令:**\n{summary}" if thinking else summary
    return result, generated_text

# 创建Gradio界面
def create_interface():
    """创建Gradio界面"""
    # 启动时就初始化模型
    print("🚀 正在启动并初始化模型...")
    initialize_model()
    print("✅ 模型初始化完成，WebUI 准备就绪！")
    
    with gr.Blocks(title="Kimi-VL Demo WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌟 Kimi-VL A3B Thinking WebUI")
        gr.Markdown("基于 Kimi-VL-A3B-Thinking-2506 模型的多模态AI助手")
        
        with gr.Tabs():
            # Tab 1: 单图像分析
            with gr.TabItem("🖼️ 图像分析"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="上传图片")
                        question_input = gr.Textbox(
                            label="问题",
                            placeholder="请输入你的问题...",
                            value="这是什么种类的猫？请用一个词回答。"
                        )
                        image_btn = gr.Button("🔍 分析图像", variant="primary")
                        
                        # 示例
                        gr.Examples(
                            examples=[
                                ["examples/demo1_cat.jpeg", "这是什么种类的猫？请用一个词回答。"],
                                ["examples/demo2_child.jpg", "孩子的鞋子和裙子是什么颜色？请以JSON格式回答。"],
                                ["examples/demo3_mathvista.jpg", "在Tiny类别中，哪个模型的语义标签准确率(%)最高？"],
                                ["examples/demo4_math_puzzle.jpg", "数字1,3,4,5和7中有四个被填入方框中，使计算正确。哪个数字没有被使用？"]
                            ],
                            inputs=[image_input, question_input]
                        )
                    
                    with gr.Column():
                        image_output = gr.Textbox(label="分析结果", lines=15, max_lines=30, show_copy_button=True)
                        image_raw = gr.Textbox(label="原始输出", lines=10, max_lines=20, visible=False)
            
            # Tab 2: PDF分析
            with gr.TabItem("📄 PDF分析"):
                with gr.Row():
                    with gr.Column():
                        pdf_input = gr.File(label="上传PDF文件", file_types=[".pdf"])
                        pdf_question = gr.Textbox(
                            label="问题",
                            placeholder="请输入你的问题...",
                            value="谁是这个基准测试的最先进方法，请分析其性能表现？"
                        )
                        pdf_btn = gr.Button("📖 分析PDF", variant="primary")
                        
                        # 示例
                        gr.Examples(
                            examples=[
                                ["examples/demo6_sample_paper.pdf", "谁是这个基准测试的最先进方法，请分析其性能表现？"],
                            ],
                            inputs=[pdf_input, pdf_question]
                        )
                    
                    with gr.Column():
                        pdf_output = gr.Textbox(label="分析结果", lines=15, max_lines=30, show_copy_button=True)
                        pdf_raw = gr.Textbox(label="原始输出", lines=10, max_lines=20, visible=False)
            
            # Tab 3: 视频分析
            with gr.TabItem("🎥 视频分析"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.File(label="上传视频文件", file_types=[".mp4", ".avi", ".mov"])
                        video_preview = gr.Video(label="视频预览", visible=False)
                        video_question = gr.Textbox(
                            label="问题",
                            placeholder="请输入你的问题...",
                            value="请将这个视频分割为场景，为每个场景提供开始时间、结束时间和详细描述。"
                        )
                        video_btn = gr.Button("🎬 分析视频", variant="primary")
                        
                        # 示例
                        gr.Examples(
                            examples=[
                                ["examples/demo7_video.mp4", "请将这个视频分割为场景，为每个场景提供开始时间、结束时间和详细描述。"],
                            ],
                            inputs=[video_input, video_question]
                        )
                    
                    with gr.Column():
                        video_output = gr.Textbox(label="分析结果", lines=15, max_lines=30, show_copy_button=True)
                        video_raw = gr.Textbox(label="原始输出", lines=10, max_lines=20, visible=False)
            
            # Tab 4: GUI Agent
            with gr.TabItem("🖱️ GUI助手"):
                with gr.Row():
                    with gr.Column():
                        gui_image = gr.Image(type="pil", label="上传屏幕截图")
                        gui_task = gr.Textbox(
                            label="任务描述",
                            placeholder="请描述你想要执行的GUI操作...",
                            value="仔细检查屏幕截图，然后点击论文提交者的个人资料。"
                        )
                        gui_btn = gr.Button("🤖 生成操作", variant="primary")
                        
                        # 示例
                        gr.Examples(
                            examples=[
                                ["examples/demo5_screenshot.png", "仔细检查屏幕截图，然后点击论文提交者的个人资料。"],
                            ],
                            inputs=[gui_image, gui_task]
                        )
                    
                    with gr.Column():
                        gui_output = gr.Textbox(label="操作指令", lines=15, max_lines=30, show_copy_button=True)
                        gui_raw = gr.Textbox(label="原始输出", lines=10, max_lines=20, visible=False)
            
            # Tab 5: 设置
            with gr.TabItem("⚙️ 设置"):
                with gr.Column():
                    gr.Markdown("### 模型状态")
                    model_status = gr.Textbox(label="状态", value="✅ 模型已初始化", interactive=False)
                    init_btn = gr.Button("� 重新初始化模型")
                    
                    gr.Markdown("### 高级设置")
                    with gr.Row():
                        show_raw = gr.Checkbox(label="显示原始输出", value=False)
                        temperature = gr.Slider(0.1, 1.0, value=0.8, label="Temperature")
                    
                    gr.Markdown("### 示例文件")
                    gr.Markdown("以下示例文件已下载到 `examples/` 目录:")
                    gr.Markdown("""
                    - `demo1_cat.jpeg` - 猫咪图片 (Demo 1)
                    - `demo2_child.jpg` - 儿童图片 (Demo 2)
                    - `demo3_mathvista.jpg` - 数学视觉题目 (Demo 3)
                    - `demo4_math_puzzle.jpg` - 数学谜题 (Demo 4)
                    - `demo5_screenshot.png` - 屏幕截图 (Demo 5)
                    - `demo6_sample_paper.pdf` - 示例PDF论文 (Demo 6)
                    - `demo7_video.mp4` - 示例视频 (Demo 7)
                    - 更多额外资源文件...
                    """)
        
        # 事件绑定
        image_btn.click(
            process_single_image,
            inputs=[image_input, question_input],
            outputs=[image_output, image_raw]
        )
        
        pdf_btn.click(
            process_pdf,
            inputs=[pdf_input, pdf_question],
            outputs=[pdf_output, pdf_raw]
        )
        
        def update_video_preview(video_file):
            """更新视频预览"""
            if video_file is not None:
                return gr.update(value=video_file.name, visible=True)
            else:
                return gr.update(value=None, visible=False)
        
        video_input.change(
            update_video_preview,
            inputs=[video_input],
            outputs=[video_preview]
        )
        
        video_btn.click(
            process_video,
            inputs=[video_input, video_question],
            outputs=[video_output, video_raw]
        )
        
        gui_btn.click(
            process_gui_agent,
            inputs=[gui_image, gui_task],
            outputs=[gui_output, gui_raw]
        )
        
        def init_model_status():
            print("🔄 重新初始化模型...")
            initialize_model()
            print("✅ 模型重新初始化完成！")
            return "✅ 模型已重新初始化"
        
        init_btn.click(
            init_model_status,
            outputs=[model_status]
        )
        
        def toggle_raw_visibility(show):
            return {
                image_raw: gr.update(visible=show),
                pdf_raw: gr.update(visible=show),
                video_raw: gr.update(visible=show),
                gui_raw: gr.update(visible=show)
            }
        
        show_raw.change(
            toggle_raw_visibility,
            inputs=[show_raw],
            outputs=[image_raw, pdf_raw, video_raw, gui_raw]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
