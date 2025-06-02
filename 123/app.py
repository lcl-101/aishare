import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# TTS Model and Interface
tts_model = None
def load_tts_model():
    global tts_model
    if tts_model is None:
        tts_model = ChatterboxTTS.from_local("checkpoints/chatterbox", DEVICE)
    return tts_model

def tts_generate(text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    model = load_tts_model()
    if seed_num != 0:
        set_seed(int(seed_num))
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    return (model.sr, wav.squeeze(0).numpy())

# VC Model and Interface
vc_model = None
def load_vc_model():
    global vc_model
    if vc_model is None:
        vc_model = ChatterboxVC.from_local("checkpoints/chatterbox", DEVICE)
    return vc_model

def vc_generate(audio, target_voice_path):
    model = load_vc_model()
    wav = model.generate(
        audio, target_voice_path=target_voice_path,
    )
    return model.sr, wav.squeeze(0).numpy()

with gr.Blocks() as demo:
    with gr.Tab("TTS 文字转语音"):
        tts_text = gr.Textbox(value="What does the fox say?", label="Text to synthesize")
        tts_ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
        tts_exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
        tts_cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)
        with gr.Accordion("More options", open=False):
            tts_seed_num = gr.Number(value=0, label="Random seed (0 for random)")
            tts_temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
        tts_run_btn = gr.Button("Generate", variant="primary")
        tts_audio_output = gr.Audio(label="Output Audio")
        tts_run_btn.click(
            fn=tts_generate,
            inputs=[
                tts_text,
                tts_ref_wav,
                tts_exaggeration,
                tts_temp,
                tts_seed_num,
                tts_cfg_weight,
            ],
            outputs=tts_audio_output,
        )
    with gr.Tab("VC 语音转换"):
        vc_input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file")
        vc_target_voice = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file (if none, the default voice is used)", value=None)
        vc_run_btn = gr.Button("Convert", variant="primary")
        vc_audio_output = gr.Audio(label="Output Audio")
        vc_run_btn.click(
            fn=vc_generate,
            inputs=[vc_input_audio, vc_target_voice],
            outputs=vc_audio_output,
        )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(server_name="0.0.0.0")
