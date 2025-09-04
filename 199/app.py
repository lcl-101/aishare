import os
import time
import subprocess
import re
from pathlib import Path
from loguru import logger
from datetime import datetime
import gradio as gr
import imageio
import numpy as np
import torch
from PIL import Image
import uuid
import glob
import threading
import queue

from voyager.utils.file_utils import save_videos_grid
from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler

from moge.model.v1 import MoGeModel
from data_engine.create_input import camera_list, depth_to_world_coords_points, render_from_cameras_videos, create_video_input


def load_models(args):
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    model = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    return model


def generate_video(temp_path: str, prompt: str) -> str:
    """单卡视频生成封装

    仅修改 app.py, 用单进程替换原来的 torchrun 8 卡调用，避免 invalid device ordinal。
    增加: 权重路径自动检测、输出文件轮询等待、友好报错。
    """
    condition_path = temp_path  # 里面包含 condition/ 目录
    output_path = os.path.join(os.path.dirname(condition_path), "output")
    os.makedirs(output_path, exist_ok=True)

    # 尝试自动定位主权重 (原逻辑里被解析成 /root/ckpts/...)，这里显式覆盖避免常见路径问题
    default_weight = "ckpts/Voyager/transformers/mp_rank_00_model_states.pt"
    weight_path = default_weight if os.path.isfile(default_weight) else None
    if weight_path is None:
        # 兜底再搜一遍
        matches = glob.glob("**/mp_rank_00_model_states.pt", recursive=True)
        if matches:
            weight_path = matches[0]
    if weight_path is None:
        raise FileNotFoundError("未找到 mp_rank_00_model_states.pt, 请确认 ckpts 已解压正确")

    env = os.environ.copy()
    # 固定只用一张卡
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    # 让内部常量解析使用本地 ckpts
    env.setdefault("MODEL_BASE", "ckpts")
    env.setdefault("ALLOW_RESIZE_FOR_SP", "1")

    cmd = [
        "python", "sample_image2video.py",
        "--model", "HYVideo-T/2",
        "--input-path", condition_path,
        "--prompt", prompt,
        "--i2v-stability",
        "--infer-steps", "50",
        "--flow-reverse",
        "--flow-shift", "7.0",
        "--seed", "0",
        "--embedded-cfg-scale", "6.0",
        "--save-path", output_path,
        "--ulysses-degree", "1",
        "--ring-degree", "1",
        "--i2v-dit-weight", weight_path,
    ]

    logger.info(f"启动单进程推理: {' '.join(cmd)}")
    cmd_start = time.time()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logger.debug(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"推理进程退出码 {proc.returncode}, 日志如下:\n{proc.stdout[-4000:]}")

    # 1) 先直接扫描输出目录
    def latest_mp4():
        if not os.path.isdir(output_path):
            return None
        mp4s = [f for f in os.listdir(output_path) if f.lower().endswith('.mp4')]
        if not mp4s:
            return None
        mp4s.sort(key=lambda f: os.path.getmtime(os.path.join(output_path, f)), reverse=True)
        return os.path.join(output_path, mp4s[0])

    target_file = latest_mp4()

    # 2) 如果目录暂时没有，尝试从 stdout 中解析 “Sample save to:” 行
    if target_file is None:
        m = re.search(r"Sample save to: (.+?\.mp4)", proc.stdout)
        if m:
            guessed = m.group(1).strip()
            if os.path.isfile(guessed):
                target_file = guessed

    # 3) 仍未找到 -> 启动独立的轮询（轮询计时从这里开始，不再把前面30+分钟算进去）
    if target_file is None:
        poll_start = time.time()
        poll_timeout = 900  # 额外等 15 分钟
        poll_interval = 2
        while time.time() - poll_start < poll_timeout:
            target_file = latest_mp4()
            if target_file:
                break
            time.sleep(poll_interval)

    if target_file is None:
        raise FileNotFoundError(
            "推理结束但未找到生成的 mp4 文件; 请查看日志末尾是否有 'Sample save to:' 行,"
            f" 输出目录: {output_path}")

    logger.info(f"视频生成完成: {target_file}")
    return target_file


def create_condition(model, image_path, direction, save_path):
    image = np.array(Image.open(image_path).resize((1280, 720)))
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device="cuda:0").permute(2, 0, 1)    
    output = model.infer(image_tensor)
    depth = np.array(output['depth'].detach().cpu())
    depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
    
    Height, Width = image.shape[:2]

    intrinsics, extrinsics = camera_list(
        num_frames=1, type=direction, Width=Width, Height=Height, fx=256, fy=256
    )

    # Backproject point cloud
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map.reshape(-1, 3)
    colors = image.reshape(-1, 3)
    
    intrinsics, extrinsics = camera_list(
        num_frames=49, type=direction, Width=Width//2, Height=Height//2, fx=128, fy=128
    )
    render_list, mask_list, depth_list = render_from_cameras_videos(
        points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
    )
    
    create_video_input(
        render_list, mask_list, depth_list, os.path.join(save_path, "condition"), separate=True, 
        ref_image=image, ref_depth=depth, Width=Width, Height=Height)

    image_list = []
    for i in range(49):
        image_list.append(np.array(Image.open(os.path.join(save_path, "condition/video_input", f"render_{i:04d}.png"))))
    imageio.mimsave(os.path.join(save_path, "condition.mp4"), image_list, fps=8)
        
    return os.path.join(save_path, "condition.mp4")


def save_uploaded_image(image, save_dir="temp_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    image_path = os.path.join(save_dir, "input_image.png")
    pil_image.save(image_path)
    return image_path


def create_video_demo():
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda:0")

    def process_condition_generation(image, direction):
        temp_path = os.path.join("temp", uuid.uuid4().hex[:8])
        image_path = save_uploaded_image(image, temp_path)
        assert image_path is not None, "Please upload image"
        condition_video_path = create_condition(moge_model, image_path, direction, temp_path)
        return os.path.join(temp_path, "condition"), condition_video_path
    
    def process_video_generation_stream(temp_path, prompt):
        """带实时进度条的生成 (Gradio generator)。

        按要求仅修改 app.py，实现：
        - 解析 sample_image2video.py 的 tqdm 输出 (x/50) 推进进度
        - 在加载阶段显示 Loading models...
        - 估算 ETA
        - 结束后返回视频路径
        """
        total_steps = 50
        if not temp_path or not prompt:
            yield _progress_html(0, total_steps, status="Waiting for inputs"), None
            return

        condition_path = temp_path
        output_path = os.path.join(os.path.dirname(condition_path), "output")
        os.makedirs(output_path, exist_ok=True)

        # 权重定位
        default_weight = "ckpts/Voyager/transformers/mp_rank_00_model_states.pt"
        weight_path = default_weight if os.path.isfile(default_weight) else None
        if weight_path is None:
            matches = glob.glob("**/mp_rank_00_model_states.pt", recursive=True)
            if matches:
                weight_path = matches[0]
        if weight_path is None:
            yield _progress_html(0, total_steps, status="Weight not found"), None
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("MODEL_BASE", "ckpts")
        env.setdefault("ALLOW_RESIZE_FOR_SP", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        cmd = [
            "python", "sample_image2video.py",
            "--model", "HYVideo-T/2",
            "--input-path", condition_path,
            "--prompt", prompt,
            "--i2v-stability",
            "--infer-steps", str(total_steps),
            "--flow-reverse",
            "--flow-shift", "7.0",
            "--seed", "0",
            "--embedded-cfg-scale", "6.0",
            "--save-path", output_path,
            "--ulysses-degree", "1",
            "--ring-degree", "1",
            "--i2v-dit-weight", weight_path,
        ]

        logger.info("Streaming inference: %s", ' '.join(cmd))
        yield _progress_html(0, total_steps, status="Loading models..."), None

        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        q = queue.Queue()
        def _reader():
            try:
                for line in proc.stdout:  # type: ignore
                    q.put(line)
            finally:
                q.put(None)
        threading.Thread(target=_reader, daemon=True).start()

        start_time = time.time()
        last_step = -1
        video_path = None
        log_lines = []
        step_regex = re.compile(r"(\b|\s)(\d+)/(?:%d)" % total_steps)

        while True:
            line = q.get()
            if line is None:
                break
            log_lines.append(line)
            s = line.strip()

            # 解析保存路径
            if 'Sample save to:' in s and s.endswith('.mp4'):
                candidate = s.split('Sample save to:')[-1].strip()
                if os.path.isfile(candidate):
                    video_path = candidate

            # 解析进度 (类似 12/50)
            m = step_regex.search(s)
            if m:
                cur = int(m.group(2))
                if 0 <= cur <= total_steps and cur != last_step:
                    last_step = cur
                    eta_txt = _eta_text(start_time, cur, total_steps)
                    yield _progress_html(cur, total_steps, status=f"Sampling {cur}/{total_steps} {eta_txt}"), None

        proc.wait()
        if proc.returncode != 0:
            tail = ''.join(log_lines)[-4000:]
            logger.error("Inference failed: %s", tail)
            yield _progress_html(last_step if last_step>0 else 0, total_steps, status="Failed"), None
            return

        # 兜底扫描输出目录
        if video_path is None:
            if os.path.isdir(output_path):
                mp4s = [f for f in os.listdir(output_path) if f.lower().endswith('.mp4')]
                if mp4s:
                    mp4s.sort(key=lambda f: os.path.getmtime(os.path.join(output_path, f)), reverse=True)
                    video_path = os.path.join(output_path, mp4s[0])

        if video_path is None:
            yield _progress_html(last_step if last_step>0 else total_steps, total_steps, status="No video found"), None
            return

        yield _progress_html(total_steps, total_steps, status="Done"), video_path
    
    with gr.Blocks(title="Voyager Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ☯️ HunyuanWorld-Voyager")
        gr.Markdown("Upload an image, input description text, select movement direction, and generate exciting videos!")

        temp_path = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                direction_choice = gr.Dropdown(
                    choices=[
                        "forward", "backward", "left", "right"
                    ],
                    label="Choose Camera Movement",
                    value="forward"
                )

                condition_video_output = gr.Video(
                    label="Condition Video",
                    height=300
                )
                
                condition_btn = gr.Button(
                    "⚙️ Generate Condition",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                input_prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Please input video description",
                    lines=3
                )
                
                gr.Markdown("### 🎥 Generating Final Video")
                progress_html = gr.HTML(value=_progress_html(0,50,status="Idle"), label="Progress")
                final_video_output = gr.Video(
                    label="Generated Video", 
                    height=600
                )

                generate_btn = gr.Button(
                    "🚀 Generate Video",
                    variant="primary",
                    size="lg"
                )
        
        examples = []
        for i in range(1, 11):
            items = [os.path.join("examples", f"case{i}", "ref_image.png"), os.path.join("examples", f"case{i}", "condition.mp4")]
            with open(os.path.join("examples", f"case{i}", "prompt.txt"), "r") as f:
                prompt = f.readline()[:-1]
                items.append(prompt)
            items.append(os.path.join("examples", f"case{i}"))
            examples.append(items)

        def update_state(hidden_input):
            return str(hidden_input)

        hidden_input = gr.Textbox(visible=False)
            
        gr.Examples(
            examples=examples,
            inputs=[input_image, condition_video_output, input_prompt, hidden_input],
            outputs=[temp_path]
        )

        hidden_input.change(fn=update_state, inputs=hidden_input, outputs=temp_path)
        
        condition_btn.click(
            fn=process_condition_generation,
            inputs=[input_image, direction_choice],
            outputs=[temp_path, condition_video_output]
        )
        generate_btn.click(
            fn=process_video_generation_stream,
            inputs=[temp_path, input_prompt],
            outputs=[progress_html, final_video_output]
        )
    
    return demo


def _progress_html(cur:int, total:int, status:str=""):
    cur = max(0, min(cur, total))
    pct = int(cur/total*100) if total>0 else 0
    bar = f"<div style='width:100%;background:#222;height:14px;border-radius:4px;overflow:hidden'>" \
          f"<div style='height:100%;width:{pct}%;background:linear-gradient(90deg,#4caf50,#8bc34a);transition:width .3s'></div></div>"
    return f"<div style='font-family:monospace'>{bar}<div style='margin-top:4px'>{pct}% ({cur}/{total}) {status}</div></div>"

def _eta_text(start_time, cur_step, total):
    if cur_step<=0:
        return ""
    elapsed = time.time()-start_time
    per = elapsed/cur_step
    remain = per*(total-cur_step)
    return f"ETA {int(remain//60)}m{int(remain%60)}s"


if __name__ == "__main__":
    demo = create_video_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
