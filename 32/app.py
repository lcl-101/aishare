import os
import tempfile
import torch
from ben2 import AutoModel
from PIL import Image
import gradio as gr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained("PramaLLC/BEN2")  # repo_id
model.to(device).eval()

def preview_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        # 预览图片
        return Image.open(filepath), None
    elif ext in [".mp4", ".webm"]:
        ## 预览视频直接返回文件路径
        return None, filepath
    else:
        return None, None

def process_file(filepath):
    # 根据文件扩展名判断是图片还是视频
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        # 图片处理逻辑
        pil_image = Image.open(filepath)
        foreground = model.inference(pil_image)
        return foreground, None
    elif ext in [".mp4", ".webm"]:
        # 视频处理逻辑
        output_dir = tempfile.gettempdir()
        output_video = os.path.join(output_dir, "foreground.mp4")
        if os.path.exists(output_video):
            os.remove(output_video)
        model.segment_video(
            video_path=filepath,
            output_path=output_dir,
            fps=0,
            refine_foreground=False,
            batch=1,
            print_frames_processed=True,
            webm=False,
            rgb_value=(0, 255, 0)
        )
        return None, output_video
    else:
        return None, None

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="上传文件（图片或视频）", type="filepath")
            preview_image = gr.Image(label="文件预览 (图片)")
            preview_video = gr.Video(label="文件预览 (视频)")
        with gr.Column():
            result_image = gr.Image(label="生成结果 (图片)")
            result_video = gr.Video(label="生成结果 (视频)")
    # 文件上传后自动预览
    file_input.change(fn=preview_file, inputs=file_input, outputs=[preview_image, preview_video])
    submit = gr.Button("提交")
    submit.click(fn=process_file, inputs=file_input, outputs=[result_image, result_video])

demo.launch(server_name='0.0.0.0')
