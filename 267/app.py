import gradio as gr
from funasr import AutoModel
import os

# 本地模型路径
MODEL_DIR = "./checkpoints/Fun-ASR-Nano-2512"

# 全局模型实例
model = None


def load_model(use_vad=True):
    """加载模型（默认启用 VAD）"""
    global model
    
    if model is None:
        print("正在加载模型...")
        if use_vad:
            model = AutoModel(
                model=MODEL_DIR,
                trust_remote_code=True,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                remote_code="./model.py",
                device="cuda:0",
                disable_update=True,
            )
        else:
            model = AutoModel(
                model=MODEL_DIR,
                trust_remote_code=True,
                remote_code="./model.py",
                device="cuda:0",
                disable_update=True,
            )
        print("模型加载完成！")
    
    return model


def transcribe_audio(audio_path):
    """
    语音转文字
    
    Args:
        audio_path: 音频文件路径
    
    Returns:
        转录的文本
    """
    if audio_path is None:
        return "请上传或录制音频文件"
    
    try:
        m = load_model()
        print(f"正在识别: {audio_path}")
        res = m.generate(input=[audio_path], cache={}, batch_size=1)
        text = res[0]["text"]
        print(f"识别结果: {text}")
        return text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"转录出错: {str(e)}"


def transcribe_with_options(audio_file, audio_mic):
    """处理上传文件或麦克风录音"""
    # 优先使用上传的文件，如果没有则使用麦克风录音
    audio_path = audio_file if audio_file else audio_mic
    
    if audio_path is None:
        return "请上传音频文件或使用麦克风录音"
    
    return transcribe_audio(audio_path)


def load_example(example_name):
    """加载示例音频"""
    example_path = os.path.join(MODEL_DIR, "example", example_name)
    if os.path.exists(example_path):
        return example_path
    return None


# 创建 Gradio 界面
with gr.Blocks(title="Fun-ASR 语音识别", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Fun-ASR 语音识别系统
        
        基于 **Fun-ASR-Nano** 模型的多语言语音识别系统，支持中文、英语、日语、韩语、粤语等多种语言。
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 上传音频")
            audio_file = gr.Audio(
                label="上传音频文件",
                type="filepath",
                sources=["upload"],
            )
            
            gr.Markdown("### �� 或使用麦克风录音")
            audio_mic = gr.Audio(
                label="麦克风录音",
                type="filepath",
                sources=["microphone"],
            )
            
            transcribe_btn = gr.Button("🚀 开始转录", variant="primary", size="lg")
            
            gr.Markdown("### 📂 示例音频")
            with gr.Row():
                example_zh = gr.Button("中文", size="sm")
                example_en = gr.Button("英语", size="sm")
                example_ja = gr.Button("日语", size="sm")
                example_ko = gr.Button("韩语", size="sm")
                example_yue = gr.Button("粤语", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📝 识别结果")
            output_text = gr.Textbox(
                label="转录文本",
                placeholder="转录结果将显示在这里...",
                lines=10,
                max_lines=20,
            )
    
    # 绑定事件
    transcribe_btn.click(
        fn=transcribe_with_options,
        inputs=[audio_file, audio_mic],
        outputs=output_text,
    )
    
    # 示例按钮事件
    example_zh.click(
        fn=lambda: load_example("zh.mp3"),
        outputs=audio_file,
    )
    example_en.click(
        fn=lambda: load_example("en.mp3"),
        outputs=audio_file,
    )
    example_ja.click(
        fn=lambda: load_example("ja.mp3"),
        outputs=audio_file,
    )
    example_ko.click(
        fn=lambda: load_example("ko.mp3"),
        outputs=audio_file,
    )
    example_yue.click(
        fn=lambda: load_example("yue.mp3"),
        outputs=audio_file,
    )
    
    gr.Markdown(
        """
        ---
        ### 📖 使用说明
        
        1. **上传音频**: 点击上传区域选择本地音频文件（支持 mp3, wav, flac 等格式）
        2. **麦克风录音**: 点击麦克风按钮开始录音，再次点击停止
        3. **示例音频**: 点击语言按钮可快速加载对应语言的示例音频
        
        ### 🌍 支持语言
        
        中文、英语、日语、韩语、粤语等多种语言
        """
    )

if __name__ == "__main__":
    # 预加载模型
    print("正在预加载模型...")
    load_model(use_vad=True)
    print("模型加载完成，启动 Web 服务...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
