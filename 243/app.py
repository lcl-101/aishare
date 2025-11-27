from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import List

import gradio as gr
import torch

from dia2 import Dia2, GenerationConfig, SamplingConfig

MAX_TURNS = 10
INITIAL_TURNS = 2

CHECKPOINT_ROOT = Path(__file__).parent / "checkpoints"
MODEL_NAME = "Dia2-2B"
CHECKPOINT_DIR = CHECKPOINT_ROOT / MODEL_NAME
CONFIG_PATH = CHECKPOINT_DIR / "config.json"
WEIGHTS_PATH = CHECKPOINT_DIR / "model.safetensors"
TOKENIZER_DIR = CHECKPOINT_DIR
MIMI_DIR = CHECKPOINT_ROOT / "mimi"
MIMI_ID = str(MIMI_DIR)
WHISPER_DIR = CHECKPOINT_ROOT / "whisper-large-v3"
DEVICE_PREF = "cuda"
DTYPE_PREF = "bfloat16"

if not os.environ.get("DIA2_WHISPER_MODEL"):
	os.environ["DIA2_WHISPER_MODEL"] = str(WHISPER_DIR)


def _patch_local_loaders() -> None:
	"""Monkey-patch third-party loaders so they always use local checkpoints."""
	try:
		from transformers import MimiModel
	except Exception:  # pragma: no cover - transformers not available
		MimiModel = None
	if MimiModel is not None and not getattr(MimiModel, "_dia2_patched", False):
		_orig_from_pretrained = MimiModel.from_pretrained.__func__

		def _dia2_local_from_pretrained(
			cls,
			pretrained_model_name_or_path,
			*model_args,
			**kwargs,
		):
			path = Path(str(pretrained_model_name_or_path)).expanduser()
			if not path.exists() and MIMI_DIR.exists():
				pretrained_model_name_or_path = str(MIMI_DIR)
				path = MIMI_DIR
			if path.exists():
				kwargs.setdefault("local_files_only", True)
			return _orig_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)

		MimiModel.from_pretrained = classmethod(_dia2_local_from_pretrained)
		MimiModel._dia2_patched = True  # type: ignore[attr-defined]

	try:
		import whisper_timestamped as wts
	except Exception:  # pragma: no cover - whisper not installed
		wts = None
	if wts is not None and not getattr(wts.load_model, "_dia2_patched", False):
		_orig_load_model = wts.load_model

		def _dia2_local_whisper(name, *args, **kwargs):
			candidates = []
			if name:
				candidates.append(Path(str(name)).expanduser())
			candidates.append(WHISPER_DIR)
			env_model = os.environ.get("DIA2_WHISPER_MODEL")
			if env_model:
				candidates.append(Path(env_model).expanduser())
			for candidate in candidates:
				if isinstance(candidate, Path) and candidate.exists():
					return _orig_load_model(str(candidate), *args, **kwargs)
			return _orig_load_model(name, *args, **kwargs)

		_dia2_local_whisper._dia2_patched = True  # type: ignore[attr-defined]
		wts.load_model = _dia2_local_whisper


_patch_local_loaders()

_dia: Dia2 | None = None


def _resolve_device(name: str) -> str:
	if name.lower() == "cuda" and torch.cuda.is_available():
		return "cuda"
	return "cpu" if name.lower() == "cuda" else name


def _validate_assets() -> None:
	required = [
		CHECKPOINT_DIR,
		CONFIG_PATH,
		WEIGHTS_PATH,
		TOKENIZER_DIR,
		MIMI_DIR,
		WHISPER_DIR,
	]
	missing = [str(path) for path in required if not path.exists()]
	if missing:
		raise FileNotFoundError(
			"Missing Dia2 assets. Ensure the checkpoint is unpacked locally: "
			+ ", ".join(missing)
		)


def _get_dia() -> Dia2:
	global _dia
	if _dia is None:
		_validate_assets()
		_dia = Dia2.from_local(
			config_path=str(CONFIG_PATH),
			weights_path=str(WEIGHTS_PATH),
			tokenizer_id=str(TOKENIZER_DIR),
			mimi_id=MIMI_ID,
			device=_resolve_device(DEVICE_PREF),
			dtype=DTYPE_PREF,
		)
	return _dia


def _concat_script(turn_count: int, turn_values: List[str]) -> str:
	lines: List[str] = []
	for idx in range(min(turn_count, len(turn_values))):
		text = (turn_values[idx] or "").strip()
		if not text:
			continue
		speaker = "[S1]" if idx % 2 == 0 else "[S2]"
		lines.append(f"{speaker} {text}")
	return "\n".join(lines)


def _example_prefix(name: str) -> str | None:
	candidate = CHECKPOINT_DIR / name
	return str(candidate) if candidate.exists() else None


EXAMPLES: dict[str, dict[str, List[str] | str | None]] = {
	"Intro": {
		"turns": [
			"Hello Dia2 fans! Today we're unveiling the new open TTS model.",
			"Sounds exciting. Can you show a sample right now?",
			"Absolutely. (laughs) Just press generate.",
		],
		"voice_s1": _example_prefix("example_1.wav"),
		"voice_s2": _example_prefix("example_2.wav"),
	},
	"Customer Support": {
		"turns": [
			"Thanks for calling. How can I help you today?",
			"My parcel never arrived and it's been two weeks.",
			"I'm sorry about that. Let me check your tracking number.",
			"Appreciate it. I really need that package soon.",
		],
		"voice_s1": _example_prefix("example_1.wav"),
		"voice_s2": _example_prefix("example_2.wav"),
	},
}


def _apply_turn_visibility(count: int) -> List[gr.Update]:
	return [gr.update(visible=i < count) for i in range(MAX_TURNS)]


def _add_turn(count: int):
	count = min(count + 1, MAX_TURNS)
	return (count, *_apply_turn_visibility(count))


def _remove_turn(count: int):
	count = max(1, count - 1)
	return (count, *_apply_turn_visibility(count))


def _load_example(name: str, count: int):
	data = EXAMPLES.get(name)
	if not data:
		return (count, *_apply_turn_visibility(count), None, None)
	turns = data.get("turns", [])
	voice_s1_path = data.get("voice_s1")
	voice_s2_path = data.get("voice_s2")
	new_count = min(len(turns), MAX_TURNS)
	updates: List[gr.Update] = []
	for idx in range(MAX_TURNS):
		if idx < new_count:
			updates.append(gr.update(value=turns[idx], visible=True))
		else:
			updates.append(gr.update(value="", visible=idx < INITIAL_TURNS))
	return (new_count, *updates, voice_s1_path, voice_s2_path)


def _prepare_prefix(file_path: str | None) -> str | None:
	if not file_path:
		return None
	path = Path(file_path)
	return str(path) if path.exists() else None


def generate_audio(turn_count: int, *inputs):
	turn_values = list(inputs[:MAX_TURNS])
	voice_s1 = inputs[MAX_TURNS]
	voice_s2 = inputs[MAX_TURNS + 1]
	cfg_scale = float(inputs[MAX_TURNS + 2])
	text_temperature = float(inputs[MAX_TURNS + 3])
	audio_temperature = float(inputs[MAX_TURNS + 4])
	text_top_k = int(inputs[MAX_TURNS + 5])
	audio_top_k = int(inputs[MAX_TURNS + 6])
	include_prefix = bool(inputs[MAX_TURNS + 7])

	script = _concat_script(turn_count, turn_values)
	if not script.strip():
		raise gr.Error("Please enter at least one non-empty speaker turn.")

	dia = _get_dia()
	config = GenerationConfig(
		cfg_scale=cfg_scale,
		text=SamplingConfig(temperature=text_temperature, top_k=text_top_k),
		audio=SamplingConfig(temperature=audio_temperature, top_k=audio_top_k),
		use_cuda_graph=True,
	)
	kwargs = {
		"prefix_speaker_1": _prepare_prefix(voice_s1),
		"prefix_speaker_2": _prepare_prefix(voice_s2),
		"include_prefix": include_prefix,
	}
	buffer = io.StringIO()
	with contextlib.redirect_stdout(buffer):
		result = dia.generate(
			script,
			config=config,
			output_wav=None,
			verbose=True,
			**kwargs,
		)
	waveform = result.waveform.detach().cpu().numpy()
	sample_rate = result.sample_rate
	timestamps = result.timestamps
	log_text = buffer.getvalue().strip()
	table = [[w, round(t, 3)] for w, t in timestamps]
	return (sample_rate, waveform), table, log_text or "Generation finished."


def build_interface() -> gr.Blocks:
	with gr.Blocks(title="Dia2 TTS") as demo:
		gr.HTML("<style>.compact-turn textarea {min-height: 60px}</style>")
		gr.Markdown(
			"""## Dia2 — Local Checkpoint\nCompose dialogue, attach optional voice prompts, and generate audio fully offline."""
		)
		turn_state = gr.State(INITIAL_TURNS)
		with gr.Row(equal_height=True):
			example_dropdown = gr.Dropdown(
				choices=["(select example)"] + list(EXAMPLES.keys()),
				label="Examples",
				value="(select example)",
			)
		with gr.Row(equal_height=True):
			with gr.Column(scale=1):
				with gr.Group():
					gr.Markdown("### Script")
					controls = []
					for idx in range(MAX_TURNS):
						speaker = "[S1]" if idx % 2 == 0 else "[S2]"
						box = gr.Textbox(
							label=f"{speaker} turn {idx + 1}",
							lines=2,
							elem_classes=["compact-turn"],
							placeholder=f"Enter dialogue for {speaker}…",
							visible=idx < INITIAL_TURNS,
						)
						controls.append(box)
					with gr.Row():
						add_btn = gr.Button("Add Turn")
						remove_btn = gr.Button("Remove Turn")
				with gr.Group():
					gr.Markdown("### Voice Prompts")
					with gr.Row():
						voice_s1 = gr.File(label="[S1] voice (wav/mp3)", type="filepath")
						voice_s2 = gr.File(label="[S2] voice (wav/mp3)", type="filepath")
				with gr.Group():
					gr.Markdown("### Sampling")
					cfg_scale = gr.Slider(1.0, 8.0, value=6.0, step=0.1, label="CFG Scale")
					with gr.Group():
						gr.Markdown("#### Text Sampling")
						text_temperature = gr.Slider(0.1, 1.5, value=0.6, step=0.05, label="Text Temperature")
						text_top_k = gr.Slider(1, 200, value=50, step=1, label="Text Top-K")
					with gr.Group():
						gr.Markdown("#### Audio Sampling")
						audio_temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Audio Temperature")
						audio_top_k = gr.Slider(1, 200, value=50, step=1, label="Audio Top-K")
					include_prefix = gr.Checkbox(label="Keep prefix audio in output", value=False)
					generate_btn = gr.Button("Generate", variant="primary")
			with gr.Column(scale=1):
				gr.Markdown("### Output")
				audio_out = gr.Audio(label="Waveform", interactive=False)
				timestamps = gr.Dataframe(headers=["word", "seconds"], label="Timestamps")
				log_box = gr.Textbox(label="Logs", lines=8)

		add_btn.click(
			lambda c: _add_turn(c),
			inputs=turn_state,
			outputs=[turn_state, *controls],
		)
		remove_btn.click(
			lambda c: _remove_turn(c),
			inputs=turn_state,
			outputs=[turn_state, *controls],
		)
		example_dropdown.change(
			lambda name, c: _load_example(name, c),
			inputs=[example_dropdown, turn_state],
			outputs=[turn_state, *controls, voice_s1, voice_s2],
		)

		generate_btn.click(
			generate_audio,
			inputs=[
				turn_state,
				*controls,
				voice_s1,
				voice_s2,
				cfg_scale,
				text_temperature,
				audio_temperature,
				text_top_k,
				audio_top_k,
				include_prefix,
			],
			outputs=[audio_out, timestamps, log_box],
		)
	return demo


if __name__ == "__main__":
	app = build_interface()
	app.queue(default_concurrency_limit=1)
	app.launch(share=False, server_name="0.0.0.0")
