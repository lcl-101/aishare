import gradio as gr
import tempfile
import shutil
import os
import torch
from omegaconf import OmegaConf
from inference import main as inference_main

# 推理包装函数
def run_inference(image, video):
    # 创建临时目录保存上传文件和输出
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "input.png")
        video_path = os.path.join(tmpdir, "input.mp4")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        # 保存上传的图片和视频
        image.save(image_path)
        shutil.copy(video, video_path)
        # 加载配置
        cfg = OmegaConf.load("config/hunyuan-portrait.yaml")
        cfg.output_dir = output_dir
        # 构造参数对象
        class Args:
            pass
        args = Args()
        args.video_path = video_path
        args.image_path = image_path
        # 推理
        inference_main(cfg, args)
        # 查找输出视频
        files = os.listdir(output_dir)
        mp4s = [f for f in files if f.endswith('.mp4')]
        if not mp4s:
            return None
        result_path = os.path.join(output_dir, mp4s[0])
        import time
        import subprocess
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = int(time.time())
        final_result_path = os.path.join(results_dir, f"webui_result_{timestamp}.mp4")
        # 判断上传视频是否有音频轨道
        ffprobe_cmd = [
            "ffprobe", "-i", video, "-show_streams", "-select_streams", "a", "-loglevel", "error"
        ]
        probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        has_audio = "codec_type=audio" in probe_result.stdout
        if has_audio:
            # 提取音频
            audio_path = os.path.join(tmpdir, "audio.aac")
            subprocess.run([
                "ffmpeg", "-y", "-i", video, "-vn", "-acodec", "copy", audio_path
            ], check=True)
            # 合成音视频
            subprocess.run([
                "ffmpeg", "-y", "-i", result_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", final_result_path
            ], check=True)
        else:
            shutil.copy(result_path, final_result_path)
        print("最终返回视频路径:", final_result_path, "文件大小:", os.path.getsize(final_result_path))
        return final_result_path

with gr.Blocks() as demo:
    gr.Markdown("# Hunyuan Portrait WebUI\n上传图片和视频，生成人像驱动视频")
    with gr.Row():
        image_input = gr.Image(type="pil", label="源图片")
        video_input = gr.Video(label="驱动视频")  # 移除 type 参数
    run_btn = gr.Button("生成")
    output_video = gr.Video(label="生成结果")
    run_btn.click(fn=run_inference, inputs=[image_input, video_input], outputs=output_video)

demo.launch(server_name="0.0.0.0")
