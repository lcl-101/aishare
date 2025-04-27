import gradio as gr
from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf

# 初始化模型
model = KimiAudio(
    model_path="checkpoints/Kimi-Audio-7B-Instruct",
    load_detokenizer=True,
)

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

def asr_inference(audio):
    if audio is None:
        return "请上传音频"
    audio_path = "temp_input.wav"
    if os.path.exists(audio_path):
        os.remove(audio_path)
    sf.write(audio_path, audio[1], audio[0])
    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
        {"role": "user", "message_type": "audio", "content": audio_path},
    ]
    _, text = model.generate(messages, **sampling_params, output_type="text")
    return text

def audio_dialog_inference(audio):
    if audio is None:
        return "请上传音频"
    audio_path = "temp_input_dialog.wav"
    if os.path.exists(audio_path):
        os.remove(audio_path)
    sf.write(audio_path, audio[1], audio[0])
    messages = [
        {
            "role": "user",
            "message_type": "audio",
            "content": audio_path,
        }
    ]
    wav, text = model.generate(messages, **sampling_params, output_type="both")
    # 返回文本和音频
    output_audio_path = "temp_output_dialog.wav"
    sf.write(output_audio_path, wav.detach().cpu().view(-1).numpy(), 24000)
    return text, output_audio_path

with gr.Blocks() as demo:
    gr.Markdown("# Kimi-Audio WebUI")
    with gr.Tab("语音转写"):
        with gr.Row():
            audio_input_asr = gr.Audio(type="numpy", label="上传音频")
        with gr.Row():
            submit_btn_asr = gr.Button("提交")
        with gr.Row():
            text_output_asr = gr.Textbox(label="识别结果")
        submit_btn_asr.click(
            asr_inference,
            inputs=[audio_input_asr],
            outputs=[text_output_asr]
        )
    with gr.Tab("语音对话"):
        with gr.Row():
            audio_input_dialog = gr.Audio(type="numpy", label="上传音频")
        with gr.Row():
            submit_btn_dialog = gr.Button("提交")
        with gr.Row():
            text_output_dialog = gr.Textbox(label="对话结果")
            audio_output_dialog = gr.Audio(label="返回音频")
        submit_btn_dialog.click(
            audio_dialog_inference,
            inputs=[audio_input_dialog],
            outputs=[text_output_dialog, audio_output_dialog]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
