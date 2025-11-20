import math
import os
import gradio as gr
import uuid
import torch
from datetime import timedelta
from lhotse import Recording
from lhotse.dataset import DynamicCutSampler
from nemo.collections.speechlm2 import SALM

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

SAMPLE_RATE = 16000 # Hz
MAX_AUDIO_MINUTES = 120 # wont try to transcribe if longer than this
CHUNK_SECONDS = 40.0  # max audio length seen by the model
BATCH_SIZE = 192  # for parallel transcription of audio longer than CHUNK_SECONDS


model = SALM.from_pretrained("checkpoints/canary-qwen-2.5b").bfloat16().eval().to(device)


def _ensure_audio_path(audio_input):
    """Gradio may pass strings, tuples, or dicts; normalize to a filesystem path."""

    if isinstance(audio_input, str):
        return audio_input
    if isinstance(audio_input, (list, tuple)) and audio_input:
        candidate = audio_input[0]
        if isinstance(candidate, str) and os.path.exists(candidate):
            return candidate
    if isinstance(audio_input, dict):
        for key in ("path", "name", "file"):
            candidate = audio_input.get(key)
            if isinstance(candidate, str) and os.path.exists(candidate):
                return candidate
    raise gr.Error("Could not resolve uploaded audio path; please upload a file or record again.")


def timestamp(idx: int):
    b = str(timedelta(seconds= idx      * CHUNK_SECONDS))
    e = str(timedelta(seconds=(idx + 1) * CHUNK_SECONDS))
    return f"[{b} - {e}]"


def as_batches(audio_filepath, utt_id, existing_rec=None):
    rec = existing_rec or Recording.from_file(audio_filepath, recording_id=utt_id)
    if rec.duration / 60.0 > MAX_AUDIO_MINUTES:
        raise gr.Error(
            f"This demo can transcribe up to {MAX_AUDIO_MINUTES} minutes of audio. "
            "If you wish, you may trim the audio using the Audio viewer in Step 1 "
            "(click on the scissors icon to start trimming audio)."
        )
    cut = rec.resample(SAMPLE_RATE).to_cut()
    if cut.num_channels > 1:
        cut = cut.to_mono(mono_downmix=True)
    return DynamicCutSampler(cut.cut_into_windows(CHUNK_SECONDS), max_cuts=BATCH_SIZE)


def transcribe(audio_filepath):
    if audio_filepath is None:
        raise gr.Error("Please provide some input audio: either upload an audio file or use the microphone")
    print(f"[DEBUG] transcribe() received type={type(audio_filepath)} value={audio_filepath}")
    audio_path = _ensure_audio_path(audio_filepath)
    print(f"[DEBUG] resolved audio path: {audio_path}")
    progress = gr.Progress(track_tqdm=True)
    progress(0.0, desc="Loading audio")
    utt_id = uuid.uuid4()
    pred_text = []
    pred_text_ts = []
    chunk_idx = 0
    recording = Recording.from_file(audio_path, recording_id=str(utt_id))
    total_chunks = max(1, math.ceil(recording.duration / CHUNK_SECONDS))
    progress(0.05, desc=f"Processing {total_chunks} chunks")
    for batch in as_batches(audio_path, str(utt_id), existing_rec=recording):
        audio, audio_lens = batch.load_audio(collate=True)
        with torch.inference_mode():
            output_ids = model.generate(
                prompts=[[{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}]] * len(batch),
                audios=torch.as_tensor(audio).to(device, non_blocking=True),
                audio_lens=torch.as_tensor(audio_lens).to(device, non_blocking=True),
                max_new_tokens=256,
            )
        texts = [model.tokenizer.ids_to_text(oids) for oids in output_ids.cpu()]
        for t in texts:
            pred_text.append(t)
            pred_text_ts.append(f"{timestamp(chunk_idx)} {t}\n\n")
            chunk_idx += 1
            progress(
                min(chunk_idx / total_chunks, 0.99),
                desc=f"Decoding chunk {chunk_idx}/{total_chunks}",
            )
    progress(1.0, desc="Done")
    return ''.join(pred_text_ts), ' '.join(pred_text)

def transcribe_ui(audio_filepath):
    transcript_with_ts, raw_text = transcribe(audio_filepath)
    return transcript_with_ts, raw_text


def summarize(transcript, prompt):
    if not transcript:
        raise gr.Error("请先完成转写，才能进行总结。")
    with torch.inference_mode(), model.llm.disable_adapter():
        output_ids = model.generate(
            prompts=[[{"role": "user", "content": f"{prompt}\n\n{transcript}"}]],
            max_new_tokens=1024,
        )
    ans = model.tokenizer.ids_to_text(output_ids[0].cpu())
    ans = ans.split("<|im_start|>assistant")[-1]
    if "<think>" in ans:
        ans = ans.split("<think>")[-1]
        thoughts, ans = ans.split("</think>")
    else:
        thoughts = ""
    return ans.strip(), thoughts.strip()

with gr.Blocks(
    title="NeMo Canary-Qwen-2.5B ASR",
    theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg),
) as demo:

    gr.Markdown(
        """
        ## Canary-Qwen-2.5B (Local)
        上传或录音，然后点击 **Transcribe**，结果会直接显示在下方文本框。
        """
    )

    audio_file = gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Audio",
    )

    transcribe_button = gr.Button(
        value="开始转写",
        variant="primary",
    )

    transcript_box = gr.Textbox(
        label="Transcription",
        placeholder="Transcription with timestamps will appear here.",
        lines=12,
    )

    raw_transcript_state = gr.State()

    transcribe_button.click(
        fn=transcribe_ui,
        inputs=audio_file,
        outputs=[transcript_box, raw_transcript_state],
    )

    with gr.Row():
        summarize_prompt = gr.Textbox(
            value="用中文总结一下讲了啥吧",
            label="LLM 提示",
        )
        summarize_button = gr.Button("一键总结", variant="secondary")

    summary_box = gr.Textbox(
        label="LLM Output",
        placeholder="Summary or Q&A result will appear here.",
        lines=8,
    )
    thoughts_box = gr.Textbox(
        label="Model Thoughts",
        placeholder="Internal reasoning (if available).",
        lines=4,
    )

    summarize_button.click(
        fn=summarize,
        inputs=[raw_transcript_state, summarize_prompt],
        outputs=[summary_box, thoughts_box],
    )


def main():
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()