"""Gradio WebUI for Hunyuan-MT with selectable target language (high-confidence list)."""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "checkpoints/Hunyuan-MT-7B"  # 可改成 "tencent/Hunyuan-MT-7B"

# 官方推荐推理参数
TOP_K = 20
TOP_P = 0.6
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.05

_model = None
_tokenizer = None

def load_model():
	global _model, _tokenizer
	if _model is None:
		dtype = (
			torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
			else (torch.float16 if torch.cuda.is_available() else None)
		)
		_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
		_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", dtype=dtype)
		_model.eval()
	return _tokenizer, _model


HIGH_CONF_LANGS = [
	"Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Italian", "Russian", "Arabic", "Turkish", "Vietnamese", "Thai", "Hindi", "Dutch", "Swedish"
]

PROMPT_ZH = "把下面的文本翻译成{target}，不要额外解释。\n\n{src}"
PROMPT_GENERAL = "Translate the following segment into {target}, without additional explanation.\n\n{src}"

def build_prompt(src: str, target: str) -> str:
	if target.lower().startswith("chinese"):
		return PROMPT_ZH.format(target="中文", src=src)
	return PROMPT_GENERAL.format(target=target, src=src)

def translate(text: str, target_language: str, max_new_tokens: int = 256):
	text = text.strip()
	if not text:
		return ""
	if target_language not in HIGH_CONF_LANGS:
		return f"[Error] Unsupported target language: {target_language}"
	tokenizer, model = load_model()
	prompt = build_prompt(text, target_language)
	messages = [{"role": "user", "content": prompt}]
	inputs = tokenizer.apply_chat_template(
		messages,
		tokenize=True,
		add_generation_prompt=False,
		return_tensors="pt"
	).to(model.device)
	with torch.inference_mode():
		outputs = model.generate(
			inputs,
			max_new_tokens=max_new_tokens,
			top_k=TOP_K,
			top_p=TOP_P,
			temperature=TEMPERATURE,
			repetition_penalty=REPETITION_PENALTY,
			do_sample=True,
		)
	gen_tokens = outputs[0][inputs.shape[-1]:]
	return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


with gr.Blocks(title="Hunyuan-MT Multi-language Translation Demo") as demo:
	gr.Markdown(
		"# Hunyuan-MT 多语种翻译 Demo\n"
		f"使用推荐参数：top_k={TOP_K}, top_p={TOP_P}, temperature={TEMPERATURE}, repetition_penalty={REPETITION_PENALTY}."
	)
	with gr.Row():
		with gr.Column():
			src_input = gr.Textbox(lines=8, label="Source Text", placeholder="输入要翻译的文本…")
			tgt_lang = gr.Dropdown(choices=HIGH_CONF_LANGS, value="Chinese", label="Target Language")
			max_new = gr.Slider(16, 512, value=256, step=16, label="Max New Tokens")
			btn = gr.Button("Translate", variant="primary")
		with gr.Column():
			output_box = gr.Textbox(lines=8, label="Translation")
	btn.click(translate, inputs=[src_input, tgt_lang, max_new], outputs=output_box)

if __name__ == "__main__":
	demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
