import os
import gradio as gr
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import tempfile
from typing import List, Optional
import re

# Global model caches
vision_model = None
vision_tokenizer = None

DEFAULT_VISION_REMOTE = "checkpoints/MiniCPM-V-4_5"


def _find_local_model(candidates: List[str]) -> Optional[str]:
	for c in candidates:
		if not c:
			continue
		p = os.path.abspath(c)
		if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
			return p
	return None


VISION_MODEL_PATH = (
	os.environ.get("MINICPM_VISION_MODEL")
	or _find_local_model([
		"MiniCPM-V-4_5",
		"./MiniCPM-V-4_5",
		"MiniCPM-V-4",
		"./MiniCPM-V-4",
	])
	or DEFAULT_VISION_REMOTE
)
OMNI_MODEL_PATH = None  # 语音模型已移除

DEVICE = os.environ.get("MINICPM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if (DEVICE.startswith("cuda") and torch.cuda.is_available()) else torch.float32

print(f"[MiniCPM] 使用 device={DEVICE}, dtype={DTYPE}, 视觉模型={VISION_MODEL_PATH}")


def load_vision_model():
	global vision_model, vision_tokenizer
	if vision_model is None:
		print(f"[MiniCPM] Loading vision model from {VISION_MODEL_PATH}")
		vision_model = AutoModel.from_pretrained(
			VISION_MODEL_PATH,
			trust_remote_code=True,
			attn_implementation='sdpa',
			torch_dtype=DTYPE,
		).eval()
		if DEVICE:
			vision_model = vision_model.to(DEVICE)
		vision_tokenizer = AutoTokenizer.from_pretrained(VISION_MODEL_PATH, trust_remote_code=True)
	return vision_model, vision_tokenizer


 # ----- Helper functions -----

def chat_vision(msgs, images=None):
	model, tokenizer = load_vision_model()
	res = model.chat(msgs=msgs, image=None, tokenizer=tokenizer)
	return res


# Single Image Tab

def single_image_fn(image, question):
	if image is None or question.strip() == "":
		return "请提供一张图片并输入问题。"
	msgs = [{'role': 'user', 'content': [image, question]}]
	return chat_vision(msgs)


# Multi Images Tab

def multi_image_fn(images, question):
	imgs = [img for img in images if img is not None]
	if len(imgs) == 0 or question.strip() == "":
		return "请至少上传一张图片并输入问题。"
	msgs = [{'role': 'user', 'content': imgs + [question]}]
	return chat_vision(msgs)


# OCR Tab

def ocr_fn(image):
	if image is None:
		return "请上传图片。"
	question = "请识别图片中的所有文字。"
	msgs = [{'role': 'user', 'content': [image, question]}]
	return chat_vision(msgs)


# Scene Text Recognition Tab (license plate)

def scene_text_fn(image):
	if image is None:
		return "请上传图片。"
	prompt = """请检测并读取图中的车牌号码，只返回车牌文本。"""
	msgs = [{'role': 'user', 'content': [prompt, image]}]
	return chat_vision(msgs)


# PDF Parse
try:
	from pdf2image import convert_from_path
except Exception:
	convert_from_path = None


def pdf_to_images(pdf_path, dpi=200):
	if convert_from_path is None:
		raise RuntimeError("pdf2image / poppler not available.")
	images = convert_from_path(pdf_path, dpi=dpi)
	return [im.convert('RGB') for im in images]


PDF_PROMPT = """
你是一个 OCR 助手。请从输入的页面图像中提取所有可见文字，并尽量保留原始版面结构，包括：\n\n- 换行与段落\n- 标题与小节\n- 表格、列表、项目符号、编号\n- 特殊字符、空格与对齐\n\n仅输出提取的文本（使用 Markdown 形式体现结构），不要添加解释或总结。
"""


def pdf_parse_fn(pdf_file):
	if pdf_file is None:
		return "请上传一个 PDF 文件。"
	try:
		# Gradio File may be a str path, a NamedString (subclass of str), or a file-like object.
		if isinstance(pdf_file, str) and os.path.exists(pdf_file):
			pdf_path = pdf_file
		elif hasattr(pdf_file, 'name') and isinstance(pdf_file.name, str) and os.path.exists(pdf_file.name):
			# Some gradio versions give an object with .name pointing to temp path
			pdf_path = pdf_file.name
		else:
			# Fallback: write bytes into a temp file
			with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
				if hasattr(pdf_file, 'read'):
					data = pdf_file.read()
				else:
					# Last resort try bytes() coercion
					data = bytes(pdf_file)
				tmp.write(data)
				pdf_path = tmp.name
		pages = pdf_to_images(pdf_path)
		msgs = [{'role': 'user', 'content': [PDF_PROMPT] + pages}]
		resp = chat_vision(msgs)
		return _clean_extracted_markdown(resp)
	except Exception as e:
		return f"解析 PDF 时出错: {e}"


def _clean_extracted_markdown(txt: str):
	"""Normalize model output for better PDF markdown rendering.

	Actions:
	- Remove stray NUL chars
	- Collapse 3+ blank lines -> 2
	- Ensure HTML <table> blocks are separated by blank lines so Gradio markdown renders them cleanly
	- Trim leading/trailing whitespace
	"""
	if not isinstance(txt, str):
		return txt
	txt = txt.replace('\x00', ' ').strip()
	# Collapse excessive blank lines
	txt = re.sub(r'\n{3,}', '\n\n', txt)
	# Surround tables with blank lines if missing (handles multi-tables)
	txt = re.sub(r'(?<!\n)\n?(<table)', '\n\n\\1', txt)
	txt = re.sub(r'(</table>)(?!\n\n)', '\1\n\n', txt)
	return txt


# Video Understanding
try:
	from decord import VideoReader, cpu
except Exception:
	VideoReader = None
	cpu = None

MAX_NUM_FRAMES = 64


def encode_video(video_path):
	"""Extract frames from a video with decord first, fallback to imageio.

	Returns list[PIL.Image]. Raises on total failure.
	"""
	from PIL import Image as _Image

	# Try decord if available
	if VideoReader is not None:
		try:
			vr = VideoReader(video_path, ctx=cpu(0))
			sample_fps = max(1, round(vr.get_avg_fps())) or 1
			frame_idx = list(range(0, len(vr), sample_fps))
			if len(frame_idx) > MAX_NUM_FRAMES:
				gap = len(frame_idx) / MAX_NUM_FRAMES
				frame_idx = [int(i * gap + gap / 2) for i in range(MAX_NUM_FRAMES)]
			frames = vr.get_batch(frame_idx).asnumpy()
			return [_Image.fromarray(f.astype('uint8')) for f in frames]
		except Exception as e:
			print(f"[Video] decord failed, will fallback. reason={e}")

	# Fallback: imageio (ffmpeg plugin)
	imageio_error = None
	try:
		import imageio
		try:
			import imageio_ffmpeg  # noqa: F401
		except Exception:
			pass
		reader = imageio.get_reader(video_path)
		meta = reader.get_meta_data()
		fps = meta.get('fps', 1) or 1
		# Aim for <= MAX_NUM_FRAMES frames uniformly
		total = meta.get('nframes', None)
		frames = []
		if total and total > MAX_NUM_FRAMES:
			# uniform sampling
			idxs = [int(i * total / MAX_NUM_FRAMES) for i in range(MAX_NUM_FRAMES)]
			for i in idxs:
				try:
					frame = reader.get_data(i)
					frames.append(_Image.fromarray(frame.astype('uint8')))
				except Exception:
					break
		else:
			for i, frame in enumerate(reader):
				frames.append(_Image.fromarray(frame.astype('uint8')))
				if len(frames) >= MAX_NUM_FRAMES:
					break
		reader.close()
		if frames:
			return frames
		imageio_error = "no frames via imageio"
	except Exception as e:
		imageio_error = str(e)

	# Fallback: OpenCV
	try:
		import cv2
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise RuntimeError("cv2 open failed")
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		idxs = []
		if frame_count > 0:
			if frame_count <= MAX_NUM_FRAMES:
				idxs = list(range(frame_count))
			else:
				gap = frame_count / MAX_NUM_FRAMES
				idxs = [int(i * gap + gap / 2) for i in range(MAX_NUM_FRAMES)]
		frames = []
		cur = 0
		want = set(idxs)
		while True and want:
			ret, frame = cap.read()
			if not ret:
				break
			if cur in want:
				frames.append(_Image.fromarray(frame[:, :, ::-1]))  # BGR->RGB
				want.remove(cur)
			cur += 1
		cap.release()
		if frames:
			return frames
		raise RuntimeError("OpenCV 未解码出任何帧")
	except Exception as e:
		raise RuntimeError(f"所有解码器失败 imageio_error={imageio_error}, opencv_error={e}")


def _resolve_file_path(file_obj, suffix=""):
	"""Return a filesystem path for a gr.File / example value.
	Accepts:
	  - str / NamedString path
	  - object with .name pointing to existing path
	  - file-like object with .read()
	Fallback: writes to a NamedTemporaryFile.
	"""
	# Direct string path
	if isinstance(file_obj, str) and os.path.exists(file_obj):
		return file_obj
	# Object with .name attribute pointing to an existing file
	if hasattr(file_obj, 'name') and isinstance(file_obj.name, str) and os.path.exists(file_obj.name):
		return file_obj.name
	# File-like (has read)
	if hasattr(file_obj, 'read'):
		data = file_obj.read()
		with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
			tmp.write(data)
			return tmp.name
	# Last resort: try bytes cast
	try:
		data = bytes(file_obj)
		with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
			tmp.write(data)
			return tmp.name
	except Exception:
		raise ValueError("不支持的文件对象类型 (video/pdf)")


def video_fn(video_file, question):
	if video_file is None or question.strip() == "":
		return "请上传一个视频并输入问题。"
	try:
		video_path = _resolve_file_path(video_file, suffix=".mp4")
		if not os.path.exists(video_path):
			return f"处理视频出错: 文件不存在 {video_path}"
		frames = encode_video(video_path)
		msgs = [{'role': 'user', 'content': frames + [question]}]
		params = {"use_image_id": False, "max_slice_nums": 2}
		model, tokenizer = load_vision_model()
		res = model.chat(msgs=msgs, image=None, tokenizer=tokenizer, **params)
		return res
	except Exception as e:
		return f"处理视频出错: {e}"


def video_preview(video_file):
	"""Return path for preview component."""
	if video_file is None:
		return None
	try:
		return _resolve_file_path(video_file, suffix=".mp4")
	except Exception:
		return None



with gr.Blocks(title="MiniCPM 视觉多模态演示") as demo:
	gr.Markdown("# MiniCPM 视觉多模态演示\n本示例聚焦视觉相关能力：单图、多图对比、OCR、车牌、PDF 解析、视频理解。")
	with gr.Tabs():
		with gr.Tab("单图理解"):
			img = gr.Image(type='pil', label="图片")
			question = gr.Textbox(label="问题", value="这张图片里的地貌类型是什么？")
			out = gr.Textbox(label="回答")
			btn = gr.Button("运行")
			btn.click(single_image_fn, inputs=[img, question], outputs=out)
			gr.Examples(
				examples=[["inference/assets/single.png", "这张图片里的地貌类型是什么？"]],
				inputs=[img, question],
				outputs=out,
				fn=single_image_fn,
				run_on_click=True,
				label="示例"
			)
		with gr.Tab("多图对比"):
			imgs_gallery = gr.Gallery(label="图片集", columns=4, height=200)
			multi_imgs_state = gr.State([])
			uploader = gr.File(file_count="multiple", file_types=['image'], label="上传图片")

			def _gather(files):
				pil_list = []
				if files:
					for f in files:
						try:
							pil = Image.open(f.name).convert('RGB')
							pil_list.append(pil)
						except Exception:
							pass
				return pil_list, pil_list

			uploader.change(_gather, inputs=uploader, outputs=[imgs_gallery, multi_imgs_state])
			question2 = gr.Textbox(label="问题", value="请比较这些图片的差异。")
			out2 = gr.Textbox(label="回答")
			run2 = gr.Button("运行")
			run2.click(multi_image_fn, inputs=[multi_imgs_state, question2], outputs=out2)

			def _load_multi_example():
				paths = ["inference/assets/multi1.png", "inference/assets/multi2.png"]
				pil_list = []
				for p in paths:
					if os.path.exists(p):
						try:
							pil_list.append(Image.open(p).convert('RGB'))
						except Exception:
							pass
				return pil_list, pil_list

			gr.Button("加载示例图片").click(_load_multi_example, outputs=[imgs_gallery, multi_imgs_state])
			gr.Markdown("示例问题已填入，可自行修改。")
		with gr.Tab("通用OCR"):
			ocr_image = gr.Image(type='pil', label="图片")
			ocr_out = gr.Textbox(label="识别结果")
			gr.Button("运行").click(ocr_fn, inputs=ocr_image, outputs=ocr_out)
			gr.Examples(
				examples=[["inference/assets/ocr.png"]],
				inputs=[ocr_image],
				outputs=ocr_out,
				fn=ocr_fn,
				run_on_click=True,
				label="示例"
			)
		with gr.Tab("车牌识别"):
			plate_img = gr.Image(type='pil', label="图片")
			plate_out = gr.Textbox(label="车牌号")
			gr.Button("运行").click(scene_text_fn, inputs=plate_img, outputs=plate_out)
			gr.Examples(
				examples=[["inference/assets/car.png"]],
				inputs=[plate_img],
				outputs=plate_out,
				fn=scene_text_fn,
				run_on_click=True,
				label="示例"
			)
		with gr.Tab("PDF解析"):
			pdf_file = gr.File(file_types=['.pdf'], label="PDF文件")
			pdf_out = gr.Markdown(label="提取结果 (Markdown)")
			gr.Button("运行").click(pdf_parse_fn, inputs=pdf_file, outputs=pdf_out)
			gr.Examples(
				examples=[["inference/assets/parse.pdf"]],
				inputs=[pdf_file],
				outputs=pdf_out,
				fn=pdf_parse_fn,
				run_on_click=True,
				label="示例"
			)
		with gr.Tab("视频理解"):
			video_file = gr.File(file_types=['.mp4', '.mov', '.avi'], label="上传视频")
			video_preview_comp = gr.Video(label="预览", autoplay=False)
			video_file.change(video_preview, inputs=video_file, outputs=video_preview_comp)
			video_q = gr.Textbox(label="问题", value="请描述该视频的主要内容。")
			video_out = gr.Textbox(label="回答")
			gr.Button("运行").click(video_fn, inputs=[video_file, video_q], outputs=video_out)
			gr.Examples(
				examples=[["inference/assets/badminton.mp4", "请描述该视频的主要内容。"]],
				inputs=[video_file, video_q],
				outputs=video_out,
				fn=video_fn,
				run_on_click=True,
				label="示例"
			)
		# 已移除语音相关 Tab

	# 已在启动时自动加载模型, 去掉手动预加载按钮


if __name__ == "__main__":
	# 启动时自动加载模型
	try:
		load_vision_model()
		print("Vision model loaded.")
	except Exception as e:
		print("视觉模型加载失败:", e)
	demo.launch(server_name="0.0.0.0", server_port=7860)
