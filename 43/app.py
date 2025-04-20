import os
import shutil
import gradio as gr
import subprocess
import yaml
import json
from PIL import Image

def get_video_name(video_file):
    if not video_file:
        return "default"
    return os.path.splitext(os.path.basename(video_file))[0]

def run_preprocessing(video_file, saved_pose_dir, saved_pose, saved_frame_dir):
    if not video_file:
        return "错误：请上传视频文件。", "default"
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, os.path.basename(video_file))
    shutil.copy(video_file, dest_path)
    
    cmd = [
        "python", "process_data.py",
        "--source_video_paths", dest_path,
        "--saved_pose_dir", saved_pose_dir,
        "--saved_pose", saved_pose,
        "--saved_frame_dir", saved_frame_dir
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"预处理失败:\n{result.stderr}", video_name
    return result.stdout if result.stdout else f"预处理完成，视频名称: {video_name}", video_name

def run_inference(cfg, input_image, video_name, debug):
    temp_cfg_path = os.path.join(os.path.dirname(cfg), "temp_infer.yaml")
    if input_image:
        upload_dir = "upload"
        images_dir = os.path.join(upload_dir, "images")
        saved_pose_dir = os.path.join(upload_dir, "saved_pose")
        saved_frames_dir = os.path.join(upload_dir, "saved_frames")
        saved_pkl_dir = os.path.join(upload_dir, "saved_pkl")
        for d in [upload_dir, images_dir, saved_pose_dir, saved_frames_dir, saved_pkl_dir]:
            os.makedirs(d, exist_ok=True)
            
        if isinstance(input_image, str):
            dest_img_path = os.path.join(images_dir, os.path.basename(input_image))
            shutil.copy(input_image, dest_img_path)
        else:
            try:
                filename = os.path.basename(input_image.name)
            except AttributeError:
                filename = "uploaded_image.png"
            dest_img_path = os.path.join(images_dir, filename)
            if hasattr(input_image, "shape"):
                input_image = Image.fromarray(input_image)
            input_image.save(dest_img_path)
        if not os.path.exists(dest_img_path):
            return "错误：图片保存失败"
        
        new_test_list = [
            2,
            os.path.abspath(dest_img_path),
            os.path.abspath(os.path.join(saved_pose_dir, video_name)),
            os.path.abspath(os.path.join(saved_frames_dir, video_name)),
            os.path.abspath(os.path.join(saved_pkl_dir, f"{video_name}.pkl")),
            14
        ]
        
        shutil.copy(cfg, temp_cfg_path)
        try:
            with open(temp_cfg_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
            if len(lines) >= 17:
                lines[9] = str(new_test_list) + '\n'
                del lines[10:17]
            with open(temp_cfg_path, 'w', encoding='utf8') as f:
                f.writelines(lines)
        except Exception as e:
            return f"配置文件修改失败: {str(e)}"
    else:
        return "错误：请上传图片文件。"
    
    try:
        cmd = ["python", "inference.py", "--cfg", temp_cfg_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)
        if result.returncode != 0:
            return f"推理失败:\n{result.stderr}"
        return result.stdout if result.stdout else "推理完成。"
    except Exception as e:
        return f"执行推理时发生错误: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## Animate-X Gradio WebUI")
    with gr.Tabs():
        with gr.TabItem("视频预处理"):
            video_control = gr.Video(label="上传并预览视频", format="mp4")
            saved_pose_dir = gr.Textbox(label="Saved Pose Dir", value="upload/saved_pkl")
            saved_pose = gr.Textbox(label="Saved Pose", value="upload/saved_pose")
            saved_frame_dir = gr.Textbox(label="Saved Frame Dir", value="upload/saved_frames")
            video_name_box = gr.Textbox(label="Video Name", value="default")
            preprocess_btn = gr.Button("运行视频预处理", variant="primary")
            preprocess_out = gr.Textbox(label="输出", lines=10)
            
            video_control.change(
                fn=get_video_name,
                inputs=[video_control],
                outputs=[video_name_box]
            )
            
            preprocess_btn.click(
                fn=run_preprocessing,
                inputs=[video_control, saved_pose_dir, saved_pose, saved_frame_dir],
                outputs=[preprocess_out, video_name_box]
            )

        with gr.TabItem("推理"):
            input_image = gr.Image(label="上传图片 (必选)", type="filepath")
            cfg = gr.Textbox(label="Config File", value="configs/Animate_X_infer.yaml")
            inference_vname = video_name_box
            debug = gr.Checkbox(label="Debug Mode", value=False)
            inference_btn = gr.Button("运行推理")
            inference_out = gr.Textbox(label="输出", lines=10)
            inference_btn.click(
                fn=run_inference,
                inputs=[cfg, input_image, inference_vname, debug],
                outputs=inference_out
            )
    
    demo.launch(server_name="0.0.0.0")
