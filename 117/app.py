import gradio as gr
import torch
import torchaudio
import os
import look2hear.models

# --- Speech Separation (TIGER Speech) ---
def separate_speech(audio_file):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    target_sr = 16000
    # Load model (cache in memory for performance)
    if not hasattr(separate_speech, "model"):
        separate_speech.model = look2hear.models.TIGER.from_pretrained("JusperLee/TIGER-speech", cache_dir="cache")
        separate_speech.model.to(device)
        separate_speech.model.eval()
    model = separate_speech.model
    # Load audio
    waveform, original_sr = torchaudio.load(audio_file)
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    audio = waveform.to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio_input = audio.unsqueeze(0).to(device)
    with torch.no_grad():
        ests_speech = model(audio_input)
    ests_speech = ests_speech.squeeze(0)
    num_speakers = ests_speech.shape[0]
    outputs = []
    for i in range(num_speakers):
        speaker_track = ests_speech[i].cpu().numpy()
        # 保证 shape 为 (samples,) 或 (1, samples)
        if speaker_track.ndim == 2 and speaker_track.shape[0] == 1:
            speaker_track = speaker_track.squeeze(0)
        if speaker_track.ndim > 1:
            speaker_track = speaker_track.reshape(-1)
        speaker_track = speaker_track.astype('float32')
        outputs.append((target_sr, speaker_track))
    # 保证输出数量为2，少于2补None
    while len(outputs) < 2:
        outputs.append(None)
    return outputs[:2]

# --- DnR Separation (TIGER DnR) ---
def separate_dnr(audio_file):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    target_sr = 44100
    if not hasattr(separate_dnr, "model"):
        separate_dnr.model = look2hear.models.TIGERDNR.from_pretrained("JusperLee/TIGER-DnR", cache_dir="cache")
        separate_dnr.model.to(device)
        separate_dnr.model.eval()
    model = separate_dnr.model
    waveform, original_sr = torchaudio.load(audio_file)
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    audio = waveform.to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    with torch.no_grad():
        all_target_dialog, all_target_effect, all_target_music = model(audio.unsqueeze(0))
    outs = []
    for t in [all_target_dialog, all_target_effect, all_target_music]:
        arr = t.squeeze(0).cpu().numpy()
        # 强制只保留一条track，保证 shape 为 (samples,)
        if arr.ndim == 2:
            arr = arr[0]
        elif arr.ndim > 2:
            arr = arr.reshape(-1)
        arr = arr.astype('float32')
        outs.append((target_sr, arr))
    return outs

with gr.Blocks() as demo:
    gr.Markdown("# TIGER Speech & DnR Separation WebUI")
    with gr.Tab("Speech Separation"):
        gr.Markdown("上传混合语音，分离说话人 (TIGER Speech)")
        speech_input = gr.Audio(type="filepath", label="上传混合语音 (WAV)")
        speech_btn = gr.Button("分离")
        speech_out1 = gr.Audio(label="Speaker 1")
        speech_out2 = gr.Audio(label="Speaker 2")
        gr.Examples([
            ["test/mix.wav"]
        ], inputs=[speech_input], label="示例音频")
        speech_btn.click(separate_speech, inputs=speech_input, outputs=[speech_out1, speech_out2])
    with gr.Tab("DnR Separation"):
        gr.Markdown("上传混合音频，分离对话、音效、音乐 (TIGER DnR)")
        dnr_input = gr.Audio(type="filepath", label="上传混合音频 (WAV)")
        dnr_btn = gr.Button("分离")
        dnr_out1 = gr.Audio(label="Dialog")
        dnr_out2 = gr.Audio(label="Effect")
        dnr_out3 = gr.Audio(label="Music")
        gr.Examples([
            ["test/test_mixture_466.wav"]
        ], inputs=[dnr_input], label="示例音频")
        dnr_btn.click(separate_dnr, inputs=dnr_input, outputs=[dnr_out1, dnr_out2, dnr_out3])

demo.launch(server_name="0.0.0.0")
