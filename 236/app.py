"""GLM-V Gradio entrypoint based on inference/trans_infer_gradio.py.

This root-level launcher hard codes the default model directory to
``checkpoints/GLM-4.5V`` so the UI can be started with ``python app.py``
without remembering any CLI switches.
"""

import copy
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

try:
    import fitz
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

import gradio as gr
import spaces
import torch
from transformers import (
    AutoProcessor,
    Glm4vForConditionalGeneration,
    Glm4vMoeForConditionalGeneration,
    TextIteratorStreamer,
)

processor = None
model = None
stop_generation = False

DEFAULT_MODEL_PATH = Path("checkpoints/GLM-4.5V-FP8")
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 7860
DEFAULT_SHARE = False
DEFAULT_MCP_SERVER = False

def load_model(model_path: str):
    global processor, model
    resolved_path = model_path
    if os.path.isdir(model_path):
        resolved_path = str(Path(model_path).resolve())
    processor = AutoProcessor.from_pretrained(resolved_path)
    if "GLM-4.5V" in resolved_path:
        model = Glm4vMoeForConditionalGeneration.from_pretrained(
            resolved_path, torch_dtype="auto", device_map="auto"
        )
    else:
        model = Glm4vForConditionalGeneration.from_pretrained(
            resolved_path, torch_dtype="auto", device_map="auto"
        )


class GLM4VModel:
    def _strip_html(self, text: str) -> str:
        return re.sub(r"<[^>]+>", "", text).strip()

    def _wrap_text(self, text: str):
        return [{"type": "text", "text": text}]

    def _pdf_to_imgs(self, pdf_path):
        doc = fitz.open(pdf_path)
        imgs = []
        for idx in range(doc.page_count):
            pix = doc.load_page(idx).get_pixmap(dpi=180)
            img_path = os.path.join(
                tempfile.gettempdir(), f"{Path(pdf_path).stem}_{idx}.png"
            )
            pix.save(img_path)
            imgs.append(img_path)
        doc.close()
        return imgs

    def _ppt_to_imgs(self, ppt_path):
        tmp_dir = tempfile.mkdtemp()
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                tmp_dir,
                ppt_path,
            ],
            check=True,
        )
        pdf_path = os.path.join(tmp_dir, Path(ppt_path).stem + ".pdf")
        return self._pdf_to_imgs(pdf_path)

    def _files_to_content(self, media):
        content = []
        for file_obj in media or []:
            suffix = Path(file_obj.name).suffix.lower()
            if suffix in [
                ".mp4",
                ".avi",
                ".mkv",
                ".mov",
                ".wmv",
                ".flv",
                ".webm",
                ".mpeg",
                ".m4v",
            ]:
                content.append({"type": "video", "url": file_obj.name})
            elif suffix in [
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".tiff",
                ".webp",
            ]:
                content.append({"type": "image", "url": file_obj.name})
            elif suffix in [".ppt", ".pptx"]:
                for img in self._ppt_to_imgs(file_obj.name):
                    content.append({"type": "image", "url": img})
            elif suffix == ".pdf":
                for img in self._pdf_to_imgs(file_obj.name):
                    content.append({"type": "image", "url": img})
        return content

    def _stream_fragment(self, buffer: str) -> str:
        think_html = ""
        if "<think>" in buffer:
            if "</think>" in buffer:
                segment = re.search(r"<think>(.*?)</think>", buffer, re.DOTALL)
                if segment:
                    think_html = (
                        "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking"  # noqa: E501
                        "</summary><div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                        + segment.group(1).strip().replace("\n", "<br>")
                        + "</div></details>"
                    )
            else:
                partial = buffer.split("<think>", 1)[1]
                think_html = (
                    "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking"
                    "</summary><div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                    + partial.replace("\n", "<br>")
                    + "</div></details>"
                )

        answer_html = ""
        if "<answer>" in buffer:
            if "</answer>" in buffer:
                segment = re.search(r"<answer>(.*?)</answer>", buffer, re.DOTALL)
                if segment:
                    answer_html = segment.group(1).strip()
            else:
                answer_html = buffer.split("<answer>", 1)[1]

        if not think_html and not answer_html:
            return self._strip_html(buffer)
        return think_html + answer_html

    def _build_messages(self, raw_history, system_prompt):
        messages = []
        if system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt.strip()}],
                }
            )

        for entry in raw_history:
            if entry["role"] == "user":
                messages.append({"role": "user", "content": entry["content"]})
            else:
                cleaned = re.sub(
                    r"<think>.*?</think>", "", entry["content"], flags=re.DOTALL
                )
                cleaned = re.sub(
                    r"<details.*?</details>", "", cleaned, flags=re.DOTALL
                )
                clean_text = self._strip_html(cleaned).strip()
                messages.append(
                    {"role": "assistant", "content": self._wrap_text(clean_text)}
                )
        return messages

    @spaces.GPU(duration=240)
    def stream_generate(self, raw_history, system_prompt):
        global stop_generation, processor, model
        stop_generation = False
        messages = self._build_messages(raw_history, system_prompt)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        streamer = TextIteratorStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        gen_kwargs = dict(
            inputs,
            max_new_tokens=8192,
            repetition_penalty=1.1,
            do_sample=True,
            top_k=2,
            temperature=None,
            top_p=1e-5,
            streamer=streamer,
        )
        generation_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        generation_thread.start()

        buffer = ""
        for token in streamer:
            if stop_generation:
                break
            buffer += token
            yield self._stream_fragment(buffer)

        generation_thread.join()


def format_display_content(content):
    if isinstance(content, list):
        text_segments = []
        file_count = 0
        for item in content:
            if item["type"] == "text":
                text_segments.append(item["text"])
            else:
                file_count += 1
        display_text = " ".join(text_segments)
        if file_count:
            return f"[{file_count} file(s) uploaded]\n{display_text}"
        return display_text
    return content


def create_display_history(raw_history):
    display_history = []
    for entry in raw_history:
        if entry["role"] == "user":
            formatted = format_display_content(entry["content"])
            display_history.append({"role": "user", "content": formatted})
        else:
            display_history.append({"role": "assistant", "content": entry["content"]})
    return display_history


glm4v = GLM4VModel()


def check_files(files):
    videos = images = ppts = pdfs = 0
    for file_obj in files or []:
        suffix = Path(file_obj.name).suffix.lower()
        if suffix in [
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".mpeg",
            ".m4v",
        ]:
            videos += 1
        elif suffix in [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        ]:
            images += 1
        elif suffix in [".ppt", ".pptx"]:
            ppts += 1
        elif suffix == ".pdf":
            pdfs += 1
    if videos > 1 or ppts > 1 or pdfs > 1:
        return False, "Only one video or one PPT or one PDF allowed"
    if images > 10:
        return False, "Maximum 10 images allowed"
    if (ppts or pdfs) and (videos or images) or (videos and images):
        return False, "Cannot mix documents, videos, and images"
    return True, ""


def chat(files, message, raw_history, system_prompt):
    global stop_generation
    stop_generation = False

    ok, error_msg = check_files(files)
    if not ok:
        raw_history.append({"role": "assistant", "content": error_msg})
        display_history = create_display_history(raw_history)
        yield display_history, copy.deepcopy(raw_history), None, ""
        return

    payload = glm4v._files_to_content(files) if files else None
    if message.strip():
        if payload is None:
            payload = glm4v._wrap_text(message.strip())
        else:
            payload.append({"type": "text", "text": message.strip()})

    user_record = {"role": "user", "content": payload if payload else message.strip()}
    raw_history = raw_history or []
    raw_history.append(user_record)

    placeholder = {"role": "assistant", "content": ""}
    raw_history.append(placeholder)

    display_history = create_display_history(raw_history)
    yield display_history, copy.deepcopy(raw_history), None, ""

    for chunk in glm4v.stream_generate(raw_history[:-1], system_prompt):
        if stop_generation:
            break
        placeholder["content"] = chunk
        display_history = create_display_history(raw_history)
        yield display_history, copy.deepcopy(raw_history), None, ""

    display_history = create_display_history(raw_history)
    yield display_history, copy.deepcopy(raw_history), None, ""


def reset():
    global stop_generation
    stop_generation = True
    time.sleep(0.1)
    return [], [], None, ""


def build_demo():
    css = """.chatbot-container .message-wrap .message{font-size:14px!important}
    details summary{cursor:pointer;font-weight:bold}
    details[open] summary{margin-bottom:10px}"""

    demo = gr.Blocks(title="GLM-4.5V Chat", theme=gr.themes.Soft(), css=css)
    with demo:
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                GLM-4.5V / GLM-4.1V Gradio UI
            </div>
            <div style="text-align: center;">
            <a href="https://huggingface.co/zai-org/GLM-4V">🤗 Model Hub</a> |
            <a href="https://github.com/zai-org/GLM-V">🌐 Github</a>
            </div>
            """
        )

        raw_history = gr.State([])

        with gr.Row():
            with gr.Column(scale=7):
                chatbox = gr.Chatbot(
                    label="Conversation",
                    type="messages",
                    height=800,
                    elem_classes="chatbot-container",
                )
                textbox = gr.Textbox(label="💭 Message")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=3):
                uploads = gr.File(
                    label="📁 Upload",
                    file_count="multiple",
                    file_types=["file"],
                    type="filepath",
                )
                gr.Markdown("Supports images / videos / PPT / PDF")
                gr.Markdown(
                    "The maximum supported input is 10 images or 1 video/PPT/PDF."
                    " During the conversation, video and images cannot be mixed."
                )
                system_prompt = gr.Textbox(label="⚙️ System Prompt", lines=6)

        gr.on(
            triggers=[send.click, textbox.submit],
            fn=chat,
            inputs=[uploads, textbox, raw_history, system_prompt],
            outputs=[chatbox, raw_history, uploads, textbox],
        )
        clear.click(reset, outputs=[chatbox, raw_history, uploads, textbox])
    return demo


def main():
    model_path = str(DEFAULT_MODEL_PATH)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model directory '{model_path}' 不存在，请确认 checkpoints 是否已同步。"
        )

    load_model(model_path)
    demo = build_demo()
    demo.launch(
        server_name=DEFAULT_SERVER_NAME,
        server_port=DEFAULT_SERVER_PORT,
        share=DEFAULT_SHARE,
        mcp_server=DEFAULT_MCP_SERVER,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
