import gradio as gr
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import tempfile
import os
import requests
import urllib.parse

# 全局配置
device = "cuda"
repo_id = "checkpoints/Voxtral-Small-24B-2507"

# 示例音频文件的URL
SAMPLE_AUDIOS = {
    "obama.mp3": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
    "bcn_weather.mp3": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    "winning_call.mp3": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
    "mary_had_lamb.mp3": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3"
}

def download_sample_audio(url, filename):
    """下载示例音频文件到本地"""
    os.makedirs("samples", exist_ok=True)
    filepath = os.path.join("samples", filename)
    
    if not os.path.exists(filepath):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename} to {filepath}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None
    
    return filepath

def download_all_samples():
    """下载所有示例音频文件"""
    downloaded_files = {}
    for filename, url in SAMPLE_AUDIOS.items():
        filepath = download_sample_audio(url, filename)
        if filepath:
            downloaded_files[filename] = filepath
    return downloaded_files

# 启动时下载所有示例文件
print("Downloading sample audio files...")
sample_files = download_all_samples()
print("Sample files ready!")

# 加载模型和处理器（只加载一次）
print("Loading Voxtral model...")
processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)
print("Model loaded successfully!")

def demo2_function(audio1_file, audio2_file, audio3_file, follow_up_question):
    """Demo2: 对话式音频分析 - 多轮对话"""
    if not audio1_file or not audio2_file or not audio3_file:
        return "请上传三个音频文件"
    
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": audio1_file,
                },
                {
                    "type": "audio",
                    "path": audio2_file,
                },
                {"type": "text", "text": "请简要描述您听到的内容。"},
            ],
        },
        {
            "role": "assistant",
            "content": "音频开始时，演讲者在芝加哥发表告别演说，回顾了他作为总统的八年时光，并向美国人民表达谢意。然后音频转向天气报告，说明巴塞罗那前一天的温度是35度，但第二天温度会降到零下20度。",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": audio3_file,
                },
                {"type": "text", "text": follow_up_question or "好的，现在请比较这个新音频与之前的音频有什么不同。"},
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(device, dtype=torch.bfloat16)
    
    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return decoded_outputs[0]

def demo3_function(text_question):
    """Demo3: 纯文本生成"""
    if not text_question:
        return "请输入问题"
    
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_question,
                },
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(device, dtype=torch.bfloat16)
    
    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return decoded_outputs[0]

def demo4_function(audio_file):
    """Demo4: 单音频分析"""
    if not audio_file:
        return "请上传音频文件"
    
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": audio_file,
                },
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(device, dtype=torch.bfloat16)
    
    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return decoded_outputs[0]

def demo5_function(audio1_file, audio2_file, audio3_file, question1, question2):
    """Demo5: 批量处理多个对话"""
    if not audio1_file or not audio2_file or not audio3_file:
        return "请上传三个音频文件"
    
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": audio1_file,
                    },
                    {
                        "type": "audio",
                        "path": audio2_file,
                    },
                    {
                        "type": "text",
                        "text": question1 or "演讲中谁在说话，讨论的是哪个城市的天气？",
                    },
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": audio3_file,
                    },
                    {"type": "text", "text": question2 or "您能告诉我这个音频的内容吗？"},
                ],
            }
        ],
    ]
    
    inputs = processor.apply_chat_template(conversations)
    inputs = inputs.to(device, dtype=torch.bfloat16)
    
    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    results = []
    for i, output in enumerate(decoded_outputs):
        results.append(f"对话 {i+1} 的回答:\n{output}")
    
    return "\n\n" + "="*80 + "\n\n".join(results)

# 创建 Gradio 界面
with gr.Blocks(title="Voxtral 多模态AI助手", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎵 Voxtral 多模态AI助手")
    gr.Markdown("这是一个集成了多种功能的语音和文本AI助手，基于Voxtral模型构建。")
    
    with gr.Tabs():
        # Demo 4 Tab - 单音频分析 (第一个)
        with gr.TabItem("单音频分析"):
            gr.Markdown("### 🎧 单个音频文件分析")
            gr.Markdown("上传一个音频文件，AI将自动分析并描述其内容。")
            
            with gr.Row():
                with gr.Column():
                    demo4_audio = gr.Audio(label="音频文件", type="filepath")
                    demo4_btn = gr.Button("🎵 分析音频", variant="primary")
                
                with gr.Column():
                    demo4_output = gr.Textbox(label="AI分析结果", lines=15, max_lines=30, show_copy_button=True)
            
            # 添加示例
            gr.Examples(
                examples=[
                    [sample_files.get("winning_call.mp3")],
                    [sample_files.get("obama.mp3")],
                    [sample_files.get("bcn_weather.mp3")],
                    [sample_files.get("mary_had_lamb.mp3")]
                ],
                inputs=[demo4_audio],
                label="📎 示例音频（点击加载）"
            )
            
            demo4_btn.click(
                demo4_function,
                inputs=demo4_audio,
                outputs=demo4_output
            )
        
        # Demo 3 Tab - 文本问答 (第二个)
        with gr.TabItem("文本问答"):
            gr.Markdown("### 📝 纯文本AI问答")
            gr.Markdown("直接向AI提问，获得详细的文本回答。")
            
            with gr.Row():
                with gr.Column():
                    demo3_question = gr.Textbox(
                        label="问题",
                        placeholder="例如：为什么AI模型应该开源？",
                        value="为什么AI模型应该开源？",
                        lines=3
                    )
                    demo3_btn = gr.Button("🤖 提问", variant="primary")
                
                with gr.Column():
                    demo3_output = gr.Textbox(label="AI回答", lines=15, max_lines=30, show_copy_button=True)
            
            # 添加示例
            gr.Examples(
                examples=[
                    ["为什么AI模型应该开源？"],
                    ["多模态AI系统有什么优势？"],
                    ["请解释AI安全和对齐的重要性。"]
                ],
                inputs=[demo3_question],
                label="📎 示例问题（点击加载）"
            )
            
            demo3_btn.click(
                demo3_function,
                inputs=demo3_question,
                outputs=demo3_output
            )
        
        # Demo 2 Tab - 对话式分析 (第三个)
        with gr.TabItem("对话式分析"):
            gr.Markdown("### 💬 多轮对话音频分析")
            gr.Markdown("上传三个音频文件，进行多轮对话式的音频内容分析。")
            
            with gr.Row():
                with gr.Column():
                    demo2_audio1 = gr.Audio(label="音频文件 1 (Obama演讲)", type="filepath")
                    demo2_audio2 = gr.Audio(label="音频文件 2 (天气报告)", type="filepath") 
                    demo2_audio3 = gr.Audio(label="音频文件 3 (对比音频)", type="filepath")
                    demo2_question = gr.Textbox(
                        label="追问问题",
                        placeholder="例如：好的，现在请比较这个新音频与之前的音频有什么不同。",
                        value="好的，现在请比较这个新音频与之前的音频有什么不同。"
                    )
                    demo2_btn = gr.Button("🗣️ 对话分析", variant="primary")
                
                with gr.Column():
                    demo2_output = gr.Textbox(label="AI回答", lines=15, max_lines=30, show_copy_button=True)
            
            # 添加示例
            gr.Examples(
                examples=[
                    [sample_files.get("obama.mp3"), sample_files.get("bcn_weather.mp3"), sample_files.get("winning_call.mp3"), "好的，现在请比较这个新音频与之前的音频有什么不同。"]
                ],
                inputs=[demo2_audio1, demo2_audio2, demo2_audio3, demo2_question],
                label="📎 示例（点击加载）"
            )
            
            demo2_btn.click(
                demo2_function,
                inputs=[demo2_audio1, demo2_audio2, demo2_audio3, demo2_question],
                outputs=demo2_output
            )
        
        # Demo 5 Tab - 批量对话处理 (第四个)
        with gr.TabItem("批量对话处理"):
            gr.Markdown("### 🔄 批量对话处理")
            gr.Markdown("同时处理多个对话，适合批量音频内容分析。")
            
            with gr.Row():
                with gr.Column():
                    demo5_audio1 = gr.Audio(label="对话1 - 音频文件 1", type="filepath")
                    demo5_audio2 = gr.Audio(label="对话1 - 音频文件 2", type="filepath")
                    demo5_question1 = gr.Textbox(
                        label="对话1 问题",
                        placeholder="例如：演讲中谁在说话，讨论的是哪个城市的天气？",
                        value="演讲中谁在说话，讨论的是哪个城市的天气？"
                    )
                    demo5_audio3 = gr.Audio(label="对话2 - 音频文件", type="filepath")
                    demo5_question2 = gr.Textbox(
                        label="对话2 问题",
                        placeholder="例如：您能告诉我这个音频的内容吗？",
                        value="您能告诉我这个音频的内容吗？"
                    )
                    demo5_btn = gr.Button("⚡ 批量处理", variant="primary")
                
                with gr.Column():
                    demo5_output = gr.Textbox(label="批量处理结果", lines=20, max_lines=40, show_copy_button=True)
            
            # 添加示例
            gr.Examples(
                examples=[
                    [
                        sample_files.get("obama.mp3"), 
                        sample_files.get("bcn_weather.mp3"), 
                        sample_files.get("winning_call.mp3"),
                        "演讲中谁在说话，讨论的是哪个城市的天气？",
                        "您能告诉我这个音频的内容吗？"
                    ]
                ],
                inputs=[demo5_audio1, demo5_audio2, demo5_audio3, demo5_question1, demo5_question2],
                label="📎 示例（点击加载）"
            )
            
            demo5_btn.click(
                demo5_function,
                inputs=[demo5_audio1, demo5_audio2, demo5_audio3, demo5_question1, demo5_question2],
                outputs=demo5_output
            )
    
    gr.Markdown("---")
    gr.Markdown("🔧 **技术栈**: Voxtral-Small-24B-2507 | Transformers | Gradio")
    gr.Markdown("💡 **提示**: 支持多种音频格式，推荐使用清晰的音频文件以获得最佳效果。")
    
    # 显示已下载的样本文件信息
    with gr.Accordion("📁 已下载的样本文件", open=False):
        sample_info = "已下载的样本音频文件:\n"
        for filename, filepath in sample_files.items():
            if filepath and os.path.exists(filepath):
                sample_info += f"✅ {filename}: {filepath}\n"
            else:
                sample_info += f"❌ {filename}: 下载失败\n"
        gr.Markdown(f"```\n{sample_info}\n```")

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False
    )
