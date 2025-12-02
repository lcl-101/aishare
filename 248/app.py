import os


def _sanitize_ipv6_in_no_proxy():
    """httpx 0.24 对未带[]的 IPv6 loopback 处理有 bug，提前剔除"""
    for key in ("no_proxy", "NO_PROXY"):
        value = os.environ.get(key)
        if not value:
            continue
        cleaned = ",".join(part for part in value.split(",") if "::" not in part)
        if cleaned:
            os.environ[key] = cleaned
        else:
            os.environ.pop(key, None)


_sanitize_ipv6_in_no_proxy()

import torch
import gradio as gr
from transformers import ARCHunyuanVideoProcessor, ARCHunyuanVideoForConditionalGeneration
from video_inference import inference

# 模型路径
MODEL_PATH = "checkpoints/ARC-Hunyuan-Video-7B"

# 全局变量存储模型和处理器
model = None
processor = None

# 任务类型说明
TASK_DESCRIPTIONS = {
    "summary": "视频摘要 - 对视频内容进行整体概括",
    "QA": "问答 - 回答关于视频内容的问题",
    "MCQ": "多选题 - 回答选择题格式的问题（需要提供选项 A/B/C/D）",
    "Grounding": "时序定位 - 定位视频中特定事件发生的时间范围",
    "segment": "章节分割 - 按时间顺序给出视频的章节摘要和对应时间点",
}


def load_model():
    """启动时自动加载模型"""
    global model, processor
    
    if model is not None:
        return
    
    print(f"正在加载模型: {MODEL_PATH}")
    
    model = (
        ARCHunyuanVideoForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .eval()
        .to("cuda")
    )
    
    processor = ARCHunyuanVideoProcessor.from_pretrained(MODEL_PATH)
    print(f"✅ 模型加载完成: {MODEL_PATH}")


def process_video(video_file, question, task_type, audio_file=None):
    """处理上传的视频"""
    global model, processor
    
    if model is None or processor is None:
        return "❌ 错误: 模型未加载！"
    
    if video_file is None:
        return "❌ 错误: 请上传视频文件！"
    
    if not question or question.strip() == "":
        return "❌ 错误: 请输入问题或指令！"
    
    try:
        # 获取上传的视频文件路径
        video_path = video_file
        audio_path = audio_file if audio_file else None
        
        print(f"处理视频: {video_path}")
        print(f"问题: {question}")
        print(f"任务类型: {task_type}")
        print(f"音频文件: {audio_path}")
        
        # 调用推理函数
        output_text = inference(
            model=model,
            processor=processor,
            question=question,
            video_path=video_path,
            audio_path=audio_path,
            task=task_type
        )
        
        return output_text
    
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理视频时出错:\n{str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def get_example_questions(task_type):
    """根据任务类型返回示例问题"""
    examples = {
        "summary": "该视频标题为[标题]\n描述视频内容.",
        "QA": "这个视频有什么亮点？",
        "MCQ": "视频中最后出现的是什么？\nA.人物\nB.动物\nC.建筑\nD.风景",
        "Grounding": "我们何时能看到特定场景或事件？",
        "segment": "请按时间顺序给出视频的章节摘要和对应时间点",
    }
    return examples.get(task_type, "请输入您的问题...")


# 创建 Gradio 界面
with gr.Blocks(title="ARC Hunyuan Video 分析系统") as demo:
    gr.Markdown(
        """
        # 🎬 ARC Hunyuan Video 分析系统
        
        这是一个基于腾讯 ARC 实验室开发的多模态视频理解模型的 Web 应用。
        支持视频摘要、问答、时序定位、章节分割等多种任务。
        
        ## 使用步骤:
        1. 上传视频文件（支持 mp4, mov, avi 等格式）
        2. 选择任务类型
        3. 输入问题或指令
        4. 可选：上传单独的音频文件
        5. 点击"开始分析"
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 任务类型选择
            gr.Markdown("### 1️⃣ 任务设置")
            task_dropdown = gr.Dropdown(
                choices=list(TASK_DESCRIPTIONS.keys()),
                value="summary",
                label="任务类型",
                interactive=True
            )
            task_description = gr.Textbox(
                label="任务说明",
                value=TASK_DESCRIPTIONS["summary"],
                interactive=False,
                lines=2
            )
            
            # 文件上传
            gr.Markdown("### 2️⃣ 文件上传")
            video_input = gr.Video(
                label="上传视频",
                sources=["upload"]
            )
            audio_input = gr.Audio(
                label="上传音频（可选，如不上传将自动从视频提取）",
                type="filepath",
                sources=["upload"]
            )
        
        with gr.Column(scale=1):
            # 问题输入
            gr.Markdown("### 3️⃣ 问题/指令")
            question_input = gr.Textbox(
                label="输入您的问题或指令",
                placeholder="请输入问题...",
                lines=5,
                value=""
            )
            
            example_btn = gr.Button("💡 填充示例问题")
            
            # 提交按钮
            submit_btn = gr.Button("🚀 开始分析", variant="primary")
            
            # 输出结果
            gr.Markdown("### 4️⃣ 分析结果")
            output_text = gr.Textbox(
                label="模型输出",
                lines=15,
                interactive=False
            )
    
    # 示例
    gr.Markdown("### 📝 使用示例")
    gr.Examples(
        examples=[
            ["summary", "该视频标题为[标题]\n描述视频内容."],
            ["QA", "这个视频的主要内容是什么？"],
            ["Grounding", "我们何时能看到特定的场景或事件？"],
            ["MCQ", "视频中最后出现的是什么？\nA.选项1\nB.选项2\nC.选项3\nD.选项4"],
            ["segment", "请按时间顺序给出视频的章节摘要和对应时间点"],
        ],
        inputs=[task_dropdown, question_input],
        label="点击加载示例"
    )
    
    # 事件处理
    def update_task_description(task):
        return TASK_DESCRIPTIONS[task]
    
    def fill_example_question(task):
        return get_example_questions(task)
    
    # 绑定事件
    task_dropdown.change(
        fn=update_task_description,
        inputs=[task_dropdown],
        outputs=[task_description]
    )
    
    example_btn.click(
        fn=fill_example_question,
        inputs=[task_dropdown],
        outputs=[question_input]
    )
    
    submit_btn.click(
        fn=process_video,
        inputs=[video_input, question_input, task_dropdown, audio_input],
        outputs=[output_text]
    )
    
    gr.Markdown(
        """
        ---
        ### 📌 注意事项
        
        - 首次加载模型需要一定时间，请耐心等待
        - 视频处理时间取决于视频长度和复杂度
        - 支持最长 300 秒的视频，超过会自动截取
        - 建议使用 GPU 运行以获得更好的性能
        - 不同任务类型需要不同的提问方式，请参考示例
        
        ### 🔧 技术信息
        
        - 模型: TencentARC ARC-Hunyuan-Video-7B
        - 框架: Transformers + Gradio
        - 支持: 视频理解、音频分析、多模态融合
        """
    )

# 启动应用
if __name__ == "__main__":
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("⚠️  警告: 未检测到 CUDA，模型将无法正常运行！")
        print("请确保您的环境配置了 GPU 支持")
    else:
        print(f"✅ 检测到 CUDA，使用 GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查模型文件是否存在
    print("\n检查模型文件...")
    if os.path.exists(MODEL_PATH):
        print(f"✅ 模型路径: {MODEL_PATH}")
    else:
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        exit(1)
    
    # 启动时自动加载模型
    print("\n正在加载模型，请稍候...")
    load_model()
    
    print("\n启动 Gradio 应用...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
