import os
import time
import torch
import gradio as gr
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

def inference(input_video_path, input_mask_path):
    print("Received input_video_path:", input_video_path)
    print("Received input_mask_path:", input_mask_path)
    
    UPLOAD_FOLDER = 'uploads'
    RESULT_FOLDER = 'results'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    
    # 定义输出路径
    priori_path = os.path.join(RESULT_FOLDER, "priori.mp4")
    output_path = os.path.join(RESULT_FOLDER, "diffueraser_result.mp4")
    
    device = get_device()
    ckpt = "2-Step"
    video_inpainting_sd = DiffuEraser(device, "weights/stable-diffusion-v1-5", "weights/sd-vae-ft-mse",
                                      "weights/diffuEraser", ckpt=ckpt)
    propainter = Propainter("weights/propainter", device=device)
    
    start_time = time.time()
    # 推理流程使用原始上传文件路径
    propainter.forward(input_video_path, input_mask_path, priori_path,
                       video_length=10, ref_stride=10, neighbor_length=10, subvideo_length=50, mask_dilation=8)
    guidance_scale = None
    video_inpainting_sd.forward(input_video_path, input_mask_path, priori_path, output_path,
                                max_img_size=960, video_length=10, mask_dilation_iter=8, guidance_scale=guidance_scale)
    end_time = time.time()
    print(f"DiffuEraser inference time: {end_time - start_time:.4f} s")
    torch.cuda.empty_cache()
    # 返回推理结果
    return output_path

# 定义页面标题和描述信息
title = "diffuEraser 测试"

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="输入视频", interactive=True)
            input_mask = gr.Video(label="遮罩视频", interactive=True)
        with gr.Column():
            output_video = gr.Video(label="结果视频")
    # 修改回调输出，只更新结果视频控件，保持上传的文件显示原始文件
    infer_button = gr.Button("推理")
    infer_button.click(fn=inference, inputs=[input_video, input_mask], outputs=output_video)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', debug=True)
