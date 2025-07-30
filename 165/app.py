import torch
import math
import numpy as np
import gradio as gr
import os

# 简化的导入，避免在没有模型时出错
try:
    import decord
    from decord import VideoReader, cpu
    from PIL import Image
    from moviepy.editor import VideoFileClip
    from pydub import AudioSegment
    import librosa
    import tempfile
    from transformers import ARCHunyuanVideoProcessor, ARCHunyuanVideoForConditionalGeneration
    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"部分依赖未安装: {e}")
    DEPS_AVAILABLE = False

# 全局变量存储模型和处理器
model = None
processor = None
processor = None


def initialize_model():
    """初始化模型"""
    global model, processor
    
    if not DEPS_AVAILABLE:
        print("❌ 依赖未安装，跳过模型加载")
        return False
    
    print("🚀 正在初始化 ARC Hunyuan Video Analysis 应用")
    
    try:
        model_path = "checkpoints/ARC-Hunyuan-Video-7B"
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型路径不存在: {model_path}")
            return False
        
        # 检查 CUDA
        if not torch.cuda.is_available():
            print("❌ CUDA 不可用，请检查GPU环境")
            return False
        
        print(f"✅ CUDA 可用，GPU: {torch.cuda.get_device_name(0)}")
        
        # 加载处理器
        print("📝 正在加载处理器...")
        processor = ARCHunyuanVideoProcessor.from_pretrained(model_path)
        print("✅ 处理器加载成功")
        
        # 加载模型
        print("🤖 正在加载模型...")
        model = ARCHunyuanVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = model.eval().to("cuda")
        print("✅ 模型加载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {str(e)}")
        return False


def calculate_frame_indices(vlen: int, fps: float, duration: float) -> list:
    frames_per_second = fps

    if duration <= 150:
        interval = 1
        intervals = [
            (int(i * interval * frames_per_second), int((i + 1) * interval * frames_per_second))
            for i in range(math.ceil(duration))
        ]
        sample_fps = 1
    else:
        num_segments = 150
        segment_duration = duration / num_segments
        intervals = [
            (int(i * segment_duration * frames_per_second), int((i + 1) * segment_duration * frames_per_second))
            for i in range(num_segments)
        ]
        sample_fps = 1 / segment_duration

    frame_indices = []
    for start, end in intervals:
        if end > vlen:
            end = vlen
        frame_indices.append((start + end) // 2)

    return frame_indices, sample_fps


def load_video_frames(video_path: str):
    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    vlen = len(video_reader)
    input_fps = video_reader.get_avg_fps()
    duration = vlen / input_fps

    frame_indices, sample_fps = calculate_frame_indices(vlen, input_fps, duration)

    return [Image.fromarray(video_reader[idx].asnumpy()) for idx in frame_indices], sample_fps


def cut_audio_with_librosa(audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    total_samples = len(audio)
    total_sec = total_samples / sr

    if total_sec <= max_total_sec:
        return audio, sr

    segment_length = total_samples // max_num_frame
    segment_samples = int(segment_sec * sr)
    segments = []
    for i in range(max_num_frame):
        start = i * segment_length
        end = min(start + segment_samples, total_samples)
        segments.append(audio[start:end])
    new_audio = np.concatenate(segments)
    return new_audio, sr


def pad_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    if len(audio) < sr:
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    return audio


def load_audio(video_path, audio_path):
    if audio_path is None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_audio:
            audio_path = temp_audio.name
            video = VideoFileClip(video_path)
            try:
                video.audio.write_audiofile(audio_path, logger=None)
                audio, sr = cut_audio_with_librosa(
                    audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000
                )
            except:
                duration = min(math.ceil(video.duration), 300)
                silent_audio = AudioSegment.silent(duration=duration * 1000)
                silent_audio.export(audio_path, format="mp3")
                audio, sr = librosa.load(audio_path, sr=16000)
    else:
        audio, sr = cut_audio_with_librosa(audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000)

    audio = pad_audio(audio, sr)
    duration = math.ceil(len(audio) / sr)

    return audio, sr, duration


def build_prompt(question: str, num_frames: int, task: str = "summary"):
    video_prefix = "<image>" * num_frames

    if task == "MCQ":
        return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer (only option index) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"
    elif task == "Grounding":
        return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer (only time range) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"
    else:  # QA、summary、segment
        return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"


def prepare_inputs(question: str, video_path: str, audio_path: str = None, task: str = "summary"):
    video_frames, sample_fps = load_video_frames(video_path)
    audio, sr, duration = load_audio(video_path, audio_path)

    # To solve mismatched duration between video and audio
    video_duration = int(len(video_frames) / sample_fps)
    audio_duration = duration

    # Truncate video frames to match audio duration
    # The audio duration will be truncated in the model
    duration = min(video_duration, audio_duration)
    if duration <= 150:
        video_frames = video_frames[: int(duration * sample_fps)]

    prompt = build_prompt(question, len(video_frames), task)

    video_inputs = {
        "video": video_frames,
        "video_metadata": {
            "fps": sample_fps,
        },
    }

    audio_inputs = {
        "audio": audio,
        "sampling_rate": sr,
        "duration": duration,
    }

    return prompt, video_inputs, audio_inputs


def inference(model, processor, question: str, video_path: str, audio_path: str = None, task: str = "summary"):
    try:
        prompt, video_inputs, audio_inputs = prepare_inputs(question, video_path, audio_path, task)
        inputs = processor(
            text=prompt,
            **video_inputs,
            **audio_inputs,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda", dtype=torch.bfloat16)
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        output_text = processor.decode(outputs[0], skip_special_tokens=True)

        return output_text
    except Exception as e:
        return f"推理过程中出现错误: {str(e)}"


def get_sample_data():
    """获取示例数据，用于演示和测试"""
    examples_data = []
    
    # First video test cases - 使用绝对路径
    video_path = os.path.abspath("examples/demo1.mp4")
    
    # Summary task
    examples_data.append([
        video_path, None, 
        "该视频标题为白金枪鱼寿司的陷阱\n描述视频内容.", 
        "summary"
    ])
    
    # Grounding task
    examples_data.append([
        video_path, None,
        "我们何时能看到一个穿制服的男人站在菊花门前?",
        "Grounding"
    ])
    
    # QA task
    examples_data.append([
        video_path, None,
        "这个视频有什么笑点？",
        "QA"
    ])
    
    # MCQ task
    examples_data.append([
        video_path, None,
        "视频中最后老板提供了什么给顾客作为赠品？\nA.纸尿裤\nB.寿司\nC.现金\nD.面巾纸",
        "MCQ"
    ])
    
    # Second video test cases - 使用绝对路径
    video_path = os.path.abspath("examples/demo3.mov")
    examples_data.append([
        video_path, None,
        "请按时间顺序给出视频的章节摘要和对应时间点",
        "segment"
    ])
    
    # Third video test cases - 使用绝对路径 
    video_path = os.path.abspath("examples/demo2.mp4")
    
    # Summary task
    examples_data.append([
        video_path, None,
        "The title of the video is\nDescribe the video content.",
        "summary"
    ])
    
    # Grounding task
    examples_data.append([
        video_path, None,
        "When will we be able to see the man in the video eat the pork cutlet in the restaurant?",
        "Grounding"
    ])
    
    # QA task
    examples_data.append([
        video_path, None,
        "Why is the man dissatisfied with the pork cutlet he cooked himself at home?",
        "QA"
    ])
    
    # Multi-granularity caption task
    examples_data.append([
        video_path, None,
        "Localize video chapters with temporal boundaries and the corresponding sentence description.",
        "segment"
    ])
    
    return examples_data


def create_demo():
    """创建 Gradio 界面"""
    
    # 检查模型是否已加载
    global model, processor
    model_status = "✅ 已就绪" if (model is not None and processor is not None) else "❌ 未加载"
    
    with gr.Blocks(title="ARC Hunyuan Video Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # 🎥 ARC Hunyuan Video Analysis
            
            基于 ARC-Hunyuan-Video-7B 模型的视频理解和分析工具
            
            **模型状态**: {model_status}
            
            ## 使用说明
            1. 点击下面的示例自动填充表单
            2. 或手动上传视频文件并输入问题
            3. 点击"分析视频"开始处理
            """
        )
        
        with gr.Row():
            with gr.Column():
                # 文件上传
                video_input = gr.File(
                    label="📹 上传视频文件",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                    type="filepath"
                )
                
                # 视频预览
                video_preview = gr.Video(
                    label="📽️ 视频预览",
                    visible=False
                )
                
                audio_input = gr.File(
                    label="🎵 上传音频文件（可选）",
                    file_types=[".mp3", ".wav", ".aac", ".m4a"],
                    type="filepath"
                )
                
                # 音频预览
                audio_preview = gr.Audio(
                    label="🎧 音频预览",
                    visible=False
                )
                
                # 问题输入
                question_input = gr.Textbox(
                    label="❓ 请输入您的问题",
                    placeholder="例如：描述这个视频的内容",
                    lines=3
                )
                
                # 任务类型选择
                task_input = gr.Dropdown(
                    label="📋 选择任务类型",
                    choices=["summary", "QA", "Grounding", "MCQ", "segment"],
                    value="summary",
                    info="summary: 视频摘要, QA: 问答, Grounding: 时间定位, MCQ: 选择题, segment: 章节分析"
                )
                
                # 分析按钮
                analyze_btn = gr.Button("🔍 分析视频", variant="secondary", size="lg")
            
            with gr.Column():
                # 结果显示
                output = gr.Textbox(
                    label="📊 分析结果",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    placeholder="分析结果将在这里显示..."
                )
        
        # 示例部分
        gr.Markdown("## 📝 内置示例")
        
        try:
            examples_data = get_sample_data()
            print(f"🔧 [调试] 准备创建 Examples 组件，总共 {len(examples_data)} 个示例")
            
            # 检查示例文件是否存在
            for i, (video_path, audio_path, question, task) in enumerate(examples_data):
                file_exists = os.path.exists(video_path) if video_path else False
                print(f"🔧 [调试] 示例 {i+1}: {os.path.basename(video_path) if video_path else 'None'} | {task} | 存在: {file_exists}")
                print(f"� [调试] 完整路径: {video_path}")
            
            # 使用 Gradio 标准 Examples 控件
            examples = gr.Examples(
                examples=examples_data,
                inputs=[video_input, audio_input, question_input, task_input],
                label="📋 点击下面的示例行自动填充表单"
            )
            
            print("✅ [调试] Examples 组件创建成功")
            
            # 使用说明
            gr.Markdown("""
            💡 **使用说明**: 
            1. **点击示例行**: 直接点击上面表格中的任意一行，会自动填充所有表单字段
            2. **分析视频**: 填充后直接点击"🔍 分析视频"按钮开始分析
            
            📁 **示例视频**:
            - `demo1.mp4`: 白金枪鱼寿司相关视频（日语）
            - `demo2.mp4`: 猪排制作相关视频（英语）  
            - `demo3.mov`: 章节分析示例视频
            """)
                        
        except Exception as e:
            print(f"⚠️ 加载示例数据失败: {e}")
            import traceback
            traceback.print_exc()
            gr.Markdown("❌ 示例数据加载失败，请检查 examples 目录")
        
        # 完整的视频分析处理函数
        def process_video(video_file, audio_file, question, task):
            global model, processor
            
            if video_file is None:
                return "请上传视频文件或使用示例！"
            if not question.strip():
                return "请输入问题！"
            if model is None or processor is None:
                return "❌ 模型未加载，请重启应用！"
            
            try:
                # 获取文件路径
                video_path = video_file if isinstance(video_file, str) else video_file.name
                audio_path = None
                if audio_file:
                    audio_path = audio_file if isinstance(audio_file, str) else audio_file.name
                
                # 执行推理
                result = inference(model, processor, question, video_path, audio_path, task)
                
                return f"✅ 视频分析完成！\n\n📹 视频: {os.path.basename(video_path)}\n❓ 问题: {question}\n📋 任务: {task}\n\n📊 分析结果:\n{result}"
                
            except Exception as e:
                error_msg = f"❌ 处理视频时出现错误: {str(e)}"
                return error_msg
        
        # 视频预览更新函数
        def update_video_preview(video_file):
            if video_file is not None:
                return gr.Video(value=video_file, visible=True)
            else:
                return gr.Video(visible=False)
        
        # 音频预览更新函数
        def update_audio_preview(audio_file):
            if audio_file is not None:
                return gr.Audio(value=audio_file, visible=True)
            else:
                return gr.Audio(visible=False)
        
        # 事件绑定
        analyze_btn.click(
            fn=process_video,
            inputs=[video_input, audio_input, question_input, task_input],
            outputs=output
        )
        
        # 文件上传时更新预览
        video_input.change(
            fn=update_video_preview,
            inputs=video_input,
            outputs=video_preview
        )
        
        audio_input.change(
            fn=update_audio_preview,
            inputs=audio_input,
            outputs=audio_preview
        )
        
        # 清空按钮
        with gr.Row():
            clear_btn = gr.Button("🗑️ 清空所有", variant="stop")
            clear_btn.click(
                lambda: ("", "", None, None, "summary"),
                outputs=[question_input, output, video_input, audio_input, task_input]
            )
    
    return demo


if __name__ == "__main__":
    print("🚀 启动 ARC Hunyuan Video Analysis...")
    
    # 初始化模型
    initialize_model()
    
    # 创建并启动 Gradio 应用
    demo = create_demo()
    
    # 启动服务器
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
