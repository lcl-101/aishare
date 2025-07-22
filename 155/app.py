import gradio as gr
import torch
import llava
from peft import PeftModel
import os
import copy
import ollama

# ---------------------------------
# SINGLE-TURN MODEL SETUP
# ---------------------------------
MODEL_BASE_SINGLE = "checkpoints/audio-flamingo-3"
MODEL_BASE_THINK = os.path.join(MODEL_BASE_SINGLE, 'stage35')

# model_single = llava.load(MODEL_BASE_SINGLE, model_base=None, devices=[0])
model_single = llava.load(MODEL_BASE_SINGLE, model_base=None)
model_single = model_single.to("cuda")
model_single_copy = copy.deepcopy(model_single)

generation_config_single = model_single.default_generation_config

model_think = PeftModel.from_pretrained(
        model_single,
        MODEL_BASE_THINK,
        device_map="auto",
        torch_dtype=torch.float16,
        )

# ---------------------------------
# MULTI-TURN MODEL SETUP
# ---------------------------------
MODEL_BASE_MULTI = "checkpoints/audio-flamingo-chat"
# model_multi = llava.load(MODEL_BASE_MULTI, model_base=None, devices=[0])
model_multi = llava.load(MODEL_BASE_MULTI, model_base=None)
model_multi = model_multi.to("cuda")
generation_config_multi = model_multi.default_generation_config


# ---------------------------------
# TRANSLATION FUNCTION
# ---------------------------------
def translate_to_chinese(text):
    try:
        response = ollama.chat(
            model='qwen3:32b',
            messages=[{
                'role': 'user',
                'content': f'请将以下英文文本翻译成中文，只返回翻译结果，不要添加任何解释：\n\n{text}\n\n/no_think'
            }]
        )
        return response['message']['content']
    except Exception as e:
        return f"翻译错误: {str(e)}"

# ---------------------------------
# SINGLE-TURN INFERENCE FUNCTION
# ---------------------------------
def single_turn_infer(audio_file, prompt_text):
    try:
        sound = llava.Sound(audio_file)
        # 移除提示词中的 <sound> 标记以避免警告
        clean_prompt = prompt_text.replace('<sound>', '').strip()
        response = model_single_copy.generate_content([sound, clean_prompt], generation_config=generation_config_single)
        translation = translate_to_chinese(response)
        return response, translation
    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        return error_msg, error_msg

def think_infer(audio_file, prompt_text):
    try:
        sound = llava.Sound(audio_file)
        # 移除提示词中的 <sound> 标记以避免警告
        clean_prompt = prompt_text.replace('<sound>', '').strip()
        response = model_think.generate_content([sound, clean_prompt], generation_config=generation_config_single)
        translation = translate_to_chinese(response)
        return response, translation
    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        return error_msg, error_msg

# ---------------------------------
# MULTI-TURN INFERENCE FUNCTION
# ---------------------------------
def multi_turn_chat(user_input, audio_file, history, current_audio):
    try:
        if audio_file is not None:
            current_audio = audio_file  # Update state if a new file is uploaded

        if current_audio is None:
            return history + [("系统", "❌ 请在聊天前先上传音频文件。")], history, current_audio

        sound = llava.Sound(current_audio)
        # 移除提示词中的 <sound> 标记以避免警告
        clean_prompt = user_input.replace('<sound>', '').strip()

        response = model_multi.generate_content([sound, clean_prompt], generation_config=generation_config_multi)

        history.append((user_input, response))
        return history, history, current_audio
    except Exception as e:
        history.append((user_input, f"❌ 错误: {str(e)}"))
        return history, history, current_audio

def speech_prompt_infer(audio_prompt_file):
    try:
        sound = llava.Sound(audio_prompt_file)
        # 对于语音提示，我们只传递音频，不需要文本提示词
        response = model_multi.generate_content([sound], generation_config=generation_config_multi)
        translation = translate_to_chinese(response)
        return response, translation
    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        return error_msg, error_msg

# ---------------------------------
# INTERFACE
# ---------------------------------
with gr.Blocks(css="""
.gradio-container { 
    max-width: 100% !important; 
    width: 100% !important;
    margin: 0 !important; 
    padding: 0 !important;
}
#component-0, .gr-block.gr-box { 
    width: 100% !important; 
}
.gr-block.gr-box, .gr-column, .gr-row {
    padding: 0 !important;
    margin: 0 !important;
}
""") as demo:

    with gr.Column():
        gr.HTML("""
<div align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA/audio-flamingo/audio_flamingo_3/static/logo-no-bg.png" alt="Audio Flamingo 3 Logo" width="120" style="margin-bottom: 10px;">
  <h2><strong>音频火烈鸟 3</strong></h2>
  <p><em>推进音频智能与完全开源的大型音频语言模型</em></p>
</div>

<div align="center" style="margin-top: 10px;">
  <a href="https://arxiv.org/abs/2507.08128">
    <img src="https://img.shields.io/badge/arXiv-2503.03983-AD1C18" alt="arXiv" style="display:inline;">
  </a>
  <a href="https://research.nvidia.com/labs/adlr/AF3/">
    <img src="https://img.shields.io/badge/Demo%20page-228B22" alt="Demo Page" style="display:inline;">
  </a>
  <a href="https://github.com/NVIDIA/audio-flamingo">
    <img src="https://img.shields.io/badge/Github-Audio_Flamingo_3-9C276A" alt="GitHub" style="display:inline;">
  </a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers">
    <img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social" alt="GitHub Stars" style="display:inline;">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/nvidia/audio-flamingo-3">
    <img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/nvidia/audio-flamingo-3-chat">
    <img src="https://img.shields.io/badge/🤗-Checkpoints_(Chat)-ED5A22.svg">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/datasets/nvidia/AudioSkills">
    <img src="https://img.shields.io/badge/🤗-Dataset:_AudioSkills--XL-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/LongAudio">
    <img src="https://img.shields.io/badge/🤗-Dataset:_LongAudio--XL-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Chat">
    <img src="https://img.shields.io/badge/🤗-Dataset:_AF--Chat-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Think">
    <img src="https://img.shields.io/badge/🤗-Dataset:_AF--Think-ED5A22.svg">
  </a>
</div>
""")
    # gr.Markdown("#### NVIDIA (2025)")

    with gr.Tabs():
        # ---------------- SINGLE-TURN ----------------
        with gr.Tab("🎯 单轮推理"):
            with gr.Row():
                with gr.Column():
                    audio_input_single = gr.Audio(type="filepath", label="上传音频")
                    prompt_input_single = gr.Textbox(label="提示词", placeholder="请对音频提出问题...", lines=8)
                    btn_single = gr.Button("生成回答")

                    gr.Examples(
                        examples=[
                            ["static/emergent/audio1.wav", "狗叫声和音乐之间有什么令人惊讶的关系？"],
                            ["static/audio/audio2.wav", "请详细描述这段音频。"],
                            ["static/speech/audio3.wav", "请转录你听到的任何语音。"],
                        ],
                        inputs=[audio_input_single, prompt_input_single],
                        label="🧪 试试这些例子"
                    )

                with gr.Column():
                    output_single = gr.Textbox(label="模型回答", lines=10)
                    translated_output_single = gr.Textbox(label="中文翻译", lines=5)

            btn_single.click(fn=single_turn_infer, inputs=[audio_input_single, prompt_input_single], outputs=[output_single, translated_output_single])

        # ---------------- THINK / LONG ----------------
        with gr.Tab("🤔 思考 / 长音频"):
            with gr.Row():
                with gr.Column():
                    audio_input_think = gr.Audio(type="filepath", label="上传音频")
                    prompt_input_think = gr.Textbox(label="提示词", placeholder="要启用思考模式，请在提示词中添加：'\n请思考并推理输入的音频内容，然后再回答。'", lines=8)
                    btn_think = gr.Button("生成回答")

                    gr.Examples(
                        examples=[
                            ["static/think/audio1.wav", "音频中的两个人在做什么？请从以下选项中选择正确答案：\n(A) 一个人在演示如何使用设备\n(B) 两个人在讨论如何使用设备\n(C) 两个人在拆卸设备\n(D) 一个人在教另一个人如何使用设备\n"],
                            ["static/think/audio2.wav", "视频中的船只是在靠近还是远离？请从以下选项中选择正确答案：\n(A) 靠近\n(B) 远离\n"],
                        ],
                        inputs=[audio_input_think, prompt_input_think],
                        label="🧪 试试这些例子"
                    )

                with gr.Column():
                    output_think = gr.Textbox(label="模型回答", lines=10)
                    translated_output_think = gr.Textbox(label="中文翻译", lines=5)

            btn_think.click(fn=think_infer, inputs=[audio_input_think, prompt_input_think], outputs=[output_think, translated_output_think])

        # ---------------- MULTI-TURN CHAT ----------------
        with gr.Tab("💬 多轮对话"):
            chatbot = gr.Chatbot(label="音频聊天机器人")
            audio_input_multi = gr.Audio(type="filepath", label="上传或替换音频内容")
            user_input_multi = gr.Textbox(label="您的消息", placeholder="对音频提出问题...", lines=8)
            btn_multi = gr.Button("发送")
            history_state = gr.State([])           # Chat history
            current_audio_state = gr.State(None)   # Most recent audio file path

            btn_multi.click(
                fn=multi_turn_chat,
                inputs=[user_input_multi, audio_input_multi, history_state, current_audio_state],
                outputs=[chatbot, history_state, current_audio_state]
            )
            gr.Examples(
                examples=[
                    ["static/chat/audio1.mp3", "这首曲子感觉非常平静和内省。是什么元素让它感觉如此平静和冥想？"],
                    ["static/chat/audio2.mp3", "换个风格，这首非常有活力且富有合成感。如果我想把那首平静的民谣重新混音成接近这种风格，你有什么建议？"],
                ],
                inputs=[audio_input_multi, user_input_multi],
                label="🧪 试试这些例子"
            )

        # ---------------- SPEECH PROMPT ----------------
        with gr.Tab("🗣️ 语音提示"):
            gr.Markdown("使用您的**语音**与模型对话。")

            with gr.Row():
                with gr.Column():
                    speech_input = gr.Audio(type="filepath", label="说话或上传音频")
                    btn_speech = gr.Button("提交")
                    gr.Examples(
                        examples=[
                            ["static/voice/voice_0.mp3"],
                            ["static/voice/voice_1.mp3"],
                            ["static/voice/voice_2.mp3"],
                        ],
                        inputs=speech_input,
                        label="🧪 试试这些例子"
                    )
                with gr.Column():
                    response_box = gr.Textbox(label="模型回答", lines=10)
                    translated_response_box = gr.Textbox(label="中文翻译", lines=5)

            btn_speech.click(fn=speech_prompt_infer, inputs=speech_input, outputs=[response_box, translated_response_box])

        # ---------------- ABOUT ----------------
        with gr.Tab("📄 关于"):
            gr.Markdown("""
### 📚 概述

**音频火烈鸟 3** 是一个完全开源的最先进（SOTA）大型音频语言模型，在语音、声音和音乐的推理和理解方面取得了重大进展。AF3 引入了：

(i) AF-Whisper，一个统一的音频编码器，使用新颖的策略进行语音、声音和音乐三种模态的联合表示学习；

(ii) 灵活的按需思考，允许模型在回答前进行链式思维推理；

(iii) 多轮、多音频对话；

(iv) 长音频理解和推理（包括语音）最长可达10分钟；以及

(v) 语音到语音交互。

为了实现这些功能，我们提出了几个使用新颖策略策划的大规模训练数据集，包括AudioSkills-XL、LongAudio-XL、AF-Think和AF-Chat，并使用新颖的五阶段课程式训练策略训练AF3。仅基于开源音频数据训练，AF3在20多个（长）音频理解和推理基准测试中取得了新的SOTA结果，超越了在更大数据集上训练的开放权重和闭源模型。

**主要特性：**

💡 音频火烈鸟3具有强大的音频、音乐和语音理解能力。

💡 音频火烈鸟3支持按需思考进行链式思维推理。

💡 音频火烈鸟3支持长达10分钟的长音频和语音理解。

💡 音频火烈鸟3可以在复杂上下文中与用户进行多轮、多音频聊天。

💡 音频火烈鸟3具有语音到语音对话能力。


""")

    gr.Markdown("© 2025 NVIDIA | 使用 ❤️ 和 Gradio + PyTorch 构建")


# -----------------------
# Launch App
# -----------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        debug=True,
        inbrowser=False
    )
