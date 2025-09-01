import os
import tempfile
import traceback
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, List, Tuple

import gradio as gr

from stepaudio2 import StepAudio2
from token2wav import Token2wav


# ----------------------- Shared Utilities -----------------------

def save_tmp_audio_bytes(audio_bytes: bytes, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        return f.name


# ----------------------- Transcription Logic -----------------------

class Transcriber:
    def __init__(self, model: StepAudio2):
        self.model = model

    def transcribe(
        self,
        audio_path: Optional[str],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
    ) -> str:
        if not audio_path:
            return "请先上传音频文件。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
            {"role": "assistant", "content": None},
        ]
        try:
            _, text, _ = self.model(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )
            return text.strip()
        except Exception as e:
            return f"转写出错: {e}"


# ----------------------- Chat (Q&A / Voice) Logic -----------------------

def add_message(chatbot, history, mic_path, text_input):
    if not mic_path and not text_input:
        return chatbot, history, "输入为空"
    if text_input:
        chatbot.append({"role": "user", "content": text_input})
        history.append({"role": "human", "content": text_input})
    elif mic_path and Path(mic_path).exists():
        chatbot.append({"role": "user", "content": {"path": mic_path}})
        history.append({"role": "human", "content": [{"type": "audio", "audio": mic_path}]})
    return chatbot, history, None


def predict_response(chatbot, history, audio_model: StepAudio2, token2wav: Token2wav, prompt_wav: str, cache_dir: str):
    try:
        history.append({"role": "assistant", "content": [{"type": "text", "text": "<tts_start>"}], "eot": False})
        tokens, text, audio_tokens = audio_model(history, max_new_tokens=4096, temperature=0.7, repetition_penalty=1.05, do_sample=True)
        audio_bytes = token2wav(audio_tokens, prompt_wav)
        audio_path = save_tmp_audio_bytes(audio_bytes, cache_dir)
        chatbot.append({"role": "assistant", "content": {"path": audio_path}})
        history[-1]["content"].append({"type": "token", "token": tokens})
        history[-1]["eot"] = True
    except Exception:
        print(traceback.format_exc())
        gr.Warning("出现错误，请重试。")
    return chatbot, history


def reset_chat(system_prompt: str):
    return [], [{"role": "system", "content": system_prompt}]


# ----------------------- UI Construction -----------------------

def build_demo(model: StepAudio2, token2wav: Token2wav, args) -> gr.Blocks:
    transcriber = Transcriber(model)

    with gr.Blocks(title="Step Audio 2 多功能 Demo", delete_cache=(86400, 86400)) as demo:
        gr.Markdown("""# Step Audio 2 多功能 Demo
提供三个功能 Tab：
1. **语音转写** —— 将语音精确转成纯文本。
2. **语音问答 (Chat)** —— 语音/文本多轮对话，可返回语音回复。
3. **语音翻译** —— 跨中英文的语音到文本 / 语音翻译。

提示：若只需干净转写，请勾选“严格仅输出原始文字”；当前版本未提供逐词时间戳，可后续接入对齐工具生成字幕。""")
        with gr.Tabs():
            # --------- Transcription Tab ---------
            with gr.TabItem("语音转写"):
                with gr.Row():
                    audio_input = gr.Audio(
                        label="上传音频 (16k wav 优先)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    transcript_box = gr.Textbox(label="转写结果", lines=12)
                with gr.Accordion("高级参数", open=False):
                    system_prompt_tr = gr.Textbox(
                        label="System Prompt",
                        value=(
                            "You are an automatic speech recognition (ASR) system. "
                            "Transcribe the user's speech verbatim. Do NOT add speaker labels, time info, summaries, analysis or any extra words. Output ONLY the raw transcript text."
                        ),
                        lines=4,
                    )
                    strict_mode = gr.Checkbox(value=True, label="严格仅输出原始文字(覆盖上面提示)")
                    max_new_tokens_tr = gr.Slider(32, 1024, value=256, step=32, label="max_new_tokens")
                    temperature_tr = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="temperature")
                    repetition_penalty_tr = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="repetition_penalty")
                with gr.Row():
                    transcribe_btn = gr.Button("🔍 转写")
                    clear_tr_btn = gr.Button("🧹 清空")

                def _clean_output(text: str) -> str:
                    # 轻量清洗：去除常见的描述性前缀行（若模型仍生成）
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    drop_prefixes = ("这是第一", "这是第一个", "这是一位", "说话的内容是", "说话内容是")
                    if len(lines) > 1:
                        filtered = [l for l in lines if not any(l.startswith(p) for p in drop_prefixes)]
                        if filtered:
                            lines = filtered
                    cleaned = " ".join(lines)
                    # 去掉可能的引号包裹
                    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith('“') and cleaned.endswith('”')):
                        cleaned = cleaned[1:-1].strip()
                    return cleaned

                def do_transcribe(audio_path, sp, strict, mnt, temp, rp):
                    if strict:
                        sp = (
                            "You are a precise speech-to-text engine. Transcribe exactly what is spoken. "
                            "No speaker labels, no timestamps, no commentary, no summarization. Output ONLY the transcribed text."
                        )
                        temp = min(float(temp), 0.5)  # 降低温度更稳定
                    raw = transcriber.transcribe(audio_path, sp, int(mnt), float(temp), float(rp))
                    if strict:
                        raw = _clean_output(raw)
                    return raw

                transcribe_btn.click(
                    do_transcribe,
                    inputs=[audio_input, system_prompt_tr, strict_mode, max_new_tokens_tr, temperature_tr, repetition_penalty_tr],
                    outputs=[transcript_box],
                    concurrency_id="gpu_queue",
                )
                clear_tr_btn.click(lambda: (None, ""), outputs=[audio_input, transcript_box])
                # 转写 Tab 不再放置示例音频（示例移动到 Chat Tab）

            # --------- Chat Tab ---------
            with gr.TabItem("语音问答 / Chat"):
                system_prompt_chat = gr.Textbox(
                    label="System Prompt",
                    value=(
                        "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n"
                        "你情感细腻，善解人意，富有同理心。请用默认女声与用户交流。"
                    ),
                    lines=3,
                )
                chatbot = gr.Chatbot(elem_id="chatbot", min_height=700, type="messages")
                # 初始化为空，会在首次发送时插入 system prompt，避免把函数对象写入历史
                history_state = gr.State([])
                mic = gr.Audio(type="filepath", label="语音输入 (可选)")
                text_input = gr.Textbox(placeholder="输入文字然后提交 ...")
                with gr.Row():
                    submit_btn = gr.Button("🚀 发送")
                    regen_btn = gr.Button("🤔 重试")
                    clear_chat_btn = gr.Button("🧹 清空会话")
                gr.Examples(
                    examples=[["assets/give_me_a_brief_introduction_to_the_great_wall.wav"]],
                    inputs=[mic],
                    label="示例音频",
                )

                def _build_system_prompt(base_sp: str) -> str:
                    return base_sp

                def on_submit(chatbot_data, history_data, mic_path, text_val, system_prompt_value):
                    system_prompt_cur = _build_system_prompt(system_prompt_value)
                    # 确保 system 在索引0 并与当前选择同步
                    if not history_data or history_data[0]["role"] != "system":
                        history_data.insert(0, {"role": "system", "content": system_prompt_cur})
                    else:
                        history_data[0]["content"] = system_prompt_cur
                    chatbot_data, history_data, err = add_message(chatbot_data, history_data, mic_path, text_val)
                    if err:
                        gr.Warning(err)
                        return chatbot_data, history_data, None, None
                    chatbot_data, history_data = predict_response(chatbot_data, history_data, model, token2wav, args.prompt_wav, args.cache_dir)
                    return chatbot_data, history_data, None, None

                submit_btn.click(
                    on_submit,
                    inputs=[chatbot, history_state, mic, text_input, system_prompt_chat],
                    outputs=[chatbot, history_state, mic, text_input],
                    concurrency_id="gpu_queue",
                )

                def regenerate(chatbot_data, history_data):
                    while chatbot_data and chatbot_data[-1]["role"] == "assistant":
                        chatbot_data.pop()
                    while history_data and history_data[-1]["role"] == "assistant":
                        history_data.pop()
                    return predict_response(chatbot_data, history_data, model, token2wav, args.prompt_wav, args.cache_dir)

                regen_btn.click(
                    regenerate,
                    inputs=[chatbot, history_state],
                    outputs=[chatbot, history_state],
                    concurrency_id="gpu_queue",
                )

                def _clear_chat(sp):
                    return reset_chat(_build_system_prompt(sp))
                clear_chat_btn.click(_clear_chat, inputs=[system_prompt_chat], outputs=[chatbot, history_state])

            # --------- Speech Translation Tab ---------
            with gr.TabItem("语音翻译"):
                gr.Markdown("""### 语音翻译 (Speech Translation)
上传或录制源语音，选择目标语言，可选生成语音形式的翻译结果。
- 仅文本：一次生成翻译文本。
- 文本+语音：一次对话请求，使用 `<tts_start>` 触发语音翻译输出。
""")
                with gr.Row():
                    audio_tr = gr.Audio(label="源音频", type="filepath", sources=["upload", "microphone"])
                    with gr.Column():
                        translated_text = gr.Textbox(label="翻译结果", lines=10)
                        translated_audio = gr.Audio(label="翻译语音", interactive=False)
                with gr.Row():
                    source_lang = gr.Dropdown(["Auto", "English", "Chinese"], value="Auto", label="源语言")
                    target_lang = gr.Dropdown(["Chinese", "English"], value="Chinese", label="目标语言")
                    voice_output = gr.Checkbox(value=False, label="生成语音输出")
                with gr.Accordion("高级参数", open=False):
                    system_prompt_trs = gr.Textbox(label="System Prompt", value="You are a helpful and professional speech translation assistant.", lines=2)
                    custom_prompt = gr.Textbox(label="自定义翻译指令 (可留空)", placeholder="留空则自动生成，如: 请把下面的英语语音翻译成地道的中文，只输出翻译。")
                    max_new_tokens_trs = gr.Slider(32, 1024, value=256, step=32, label="max_new_tokens")
                    temperature_trs = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="temperature")
                    repetition_penalty_trs = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="repetition_penalty")
                with gr.Row():
                    translate_btn = gr.Button("🌐 翻译")
                    clear_trs_btn = gr.Button("🧹 清空")

                def build_instruction(src: str, tgt: str, voice: bool) -> str:
                    tgt_cn = "中文" if tgt == "Chinese" else "English"
                    if src == "Auto":
                        base = f"Please translate the following speech into {tgt}. Output only the {tgt} translation." if tgt != "Chinese" else "请把下面的语音内容翻译成自然流畅的中文，只输出翻译。"
                    else:
                        if src == "English" and tgt == "Chinese":
                            base = "请把下面的英语语音翻译成自然、准确且简洁的中文，只输出翻译。"
                        elif src == "Chinese" and tgt == "English":
                            base = "Please translate the following Chinese speech into concise, natural English. Output only the translation." 
                        else:
                            base = f"Please translate the following {src} speech into {tgt}. Output only the {tgt} translation."
                    if voice:
                        base += " Provide the translated content as speech (TTS) as well." if tgt != "Chinese" else " 并使用语音形式朗读译文。"
                    return base

                def speech_translate(audio_path, src_lang, tgt_lang, voice_out, system_prompt, custom_inst, max_new, temp, rep):
                    if not audio_path:
                        return "请先上传音频。", None
                    instruction = custom_inst.strip() if custom_inst and custom_inst.strip() else build_instruction(src_lang, tgt_lang, voice_out)
                    human_content = [
                        {"type": "text", "text": instruction},
                        {"type": "audio", "audio": audio_path},
                    ]
                    assistant_content = "<tts_start>" if voice_out else None
                    assistant_kwargs = {"eot": False} if voice_out else {}
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "human", "content": human_content},
                        {"role": "assistant", "content": assistant_content, **assistant_kwargs},
                    ]
                    try:
                        tokens, text, audio_tokens = model(
                            messages,
                            max_new_tokens=int(max_new),
                            temperature=float(temp),
                            repetition_penalty=float(rep),
                            do_sample=True,
                        )
                        audio_path_out = None
                        if voice_out and audio_tokens:
                            try:
                                audio_bytes = token2wav(audio_tokens, args.prompt_wav)
                                audio_path_out = save_tmp_audio_bytes(audio_bytes, args.cache_dir)
                            except Exception as e:
                                gr.Warning(f"语音合成失败: {e}")
                        return text.strip(), audio_path_out
                    except Exception as e:
                        return f"翻译出错: {e}", None

                translate_btn.click(
                    speech_translate,
                    inputs=[audio_tr, source_lang, target_lang, voice_output, system_prompt_trs, custom_prompt, max_new_tokens_trs, temperature_trs, repetition_penalty_trs],
                    outputs=[translated_text, translated_audio],
                    concurrency_id="gpu_queue",
                )
                clear_trs_btn.click(lambda: (None, "", None), outputs=[audio_tr, translated_text, translated_audio])
                gr.Examples(
                    examples=[["assets/give_me_a_brief_introduction_to_the_great_wall.wav"]],
                    inputs=[audio_tr],
                    label="示例音频",
                )

        gr.Markdown(
            """**提示**: 当前模型未输出逐词时间戳，若需要字幕可单独做分段+转写或集成外部对齐工具 (WhisperX/MFA)。"""
        )

    return demo


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Step-Audio-2-mini", help="模型目录")
    parser.add_argument("--server-port", type=int, default=7860, help="Server 端口")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server 绑定地址")
    parser.add_argument("--prompt-wav", type=str, default="assets/default_female.wav", help="聊天语音回复的提示音色 wav")
    parser.add_argument("--cache-dir", type=str, default="/tmp/stepaudio2", help="缓存目录")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["GRADIO_TEMP_DIR"] = args.cache_dir
    model = StepAudio2(args.model_path)
    token2wav = Token2wav(f"{args.model_path}/token2wav")
    demo = build_demo(model, token2wav, args)
    demo.queue().launch(server_port=args.server_port, server_name=args.server_name)


if __name__ == "__main__":
    main()
