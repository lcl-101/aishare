#!/usr/bin/env python
"""
Wan2.2 Animate 预处理 WebUI (第一步)
-------------------------------------------------
本页面仅实现官方命令中第 1 步的功能：

python ./wan/modules/animate/preprocess/preprocess_data.py \
	--ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
	--video_path <video> \
	--refer_path <image> \
	--save_path <output_dir> \
	--resolution_area 1280 720 \
	--retarget_flag \
	--use_flux

无需调用外部命令，而是直接在 Python 内部构建并调用 ProcessPipeline。

功能：
  * 用户可上传视频 / 图片 或 选择 examples 目录的演示文件
  * 可配置：resolution_area (宽 高)、retarget_flag、use_flux
  * 运行后输出目录，并列出生成的关键文件

后续第二步 (视频生成) 可在后面再扩展。
"""
from __future__ import annotations
import os
import gradio as gr
import shutil
import argparse
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import sys

# --- 动态修正预处理模块的导入路径 ---
# 原始 preprocess 代码使用了形如 `from pose2d import Pose2d` 的顶层导入，
# 需要把其所在目录加入 sys.path 才能在不改源码的情况下正常导入。
_PREPROCESS_DIR = os.path.join(os.path.dirname(__file__), 'wan', 'modules', 'animate', 'preprocess')
if os.path.isdir(_PREPROCESS_DIR) and _PREPROCESS_DIR not in sys.path:
	sys.path.insert(0, _PREPROCESS_DIR)

try:
	from wan.modules.animate.preprocess import ProcessPipeline  # 仍保持官方包路径导入
except ModuleNotFoundError:
	# 再尝试直接本地导入（兼容用户未来调整）
	from process_pipepline import ProcessPipeline  # type: ignore

# 第二步需要导入主模型
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
import torch


DEFAULT_CKPT_DIR = './Wan2.2-Animate-14B/process_checkpoint'
EXAMPLE_VIDEO = 'examples/wan_animate/animate/video.mp4'
EXAMPLE_IMAGE = 'examples/wan_animate/animate/image.jpeg'
REPLACE_EXAMPLE_VIDEO = 'examples/wan_animate/replace/video.mp4'
REPLACE_EXAMPLE_IMAGE = 'examples/wan_animate/replace/image.jpeg'
MODEL_CKPT_DIR = os.path.dirname(DEFAULT_CKPT_DIR.rstrip('/'))  # e.g. ./Wan2.2-Animate-14B

# 缓存生成模型：按 (device, use_relighting_lora) 组合缓存
_ANIMATE_MODELS: Dict[Tuple[str, bool], wan.WanAnimate] = {}

def get_animate_model(device: str = 'cuda:0', use_relighting_lora: bool = False) -> wan.WanAnimate:
    key = (device, bool(use_relighting_lora))
    if key not in _ANIMATE_MODELS:
        cfg = WAN_CONFIGS['animate-14B']
        device_id = int(device.split(':')[-1])
        _ANIMATE_MODELS[key] = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=MODEL_CKPT_DIR,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=False,
            use_relighting_lora=use_relighting_lora
        )
    return _ANIMATE_MODELS[key]


def build_pipeline(replace_flag: bool, use_flux: bool):
	"""根据参数构建单次使用的 ProcessPipeline。
	（当前第一步不需要 replacement，所以 replace_flag 暂固定 False，接口保留扩展点）"""
	pose2d_checkpoint_path = os.path.join(DEFAULT_CKPT_DIR, 'pose2d/vitpose_h_wholebody.onnx')
	det_checkpoint_path = os.path.join(DEFAULT_CKPT_DIR, 'det/yolov10m.onnx')
	sam2_checkpoint_path = os.path.join(DEFAULT_CKPT_DIR, 'sam2/sam2_hiera_large.pt') if replace_flag else None
	flux_kontext_path = os.path.join(DEFAULT_CKPT_DIR, 'FLUX.1-Kontext-dev') if use_flux else None
	return ProcessPipeline(det_checkpoint_path=det_checkpoint_path,
						   pose2d_checkpoint_path=pose2d_checkpoint_path,
						   sam_checkpoint_path=sam2_checkpoint_path,
						   flux_kontext_path=flux_kontext_path)


def list_outputs(output_dir: str) -> List[str]:
	files = []
	for name in ['src_pose.mp4', 'src_face.mp4', 'src_ref.png', 'src_bg.mp4', 'src_mask.mp4', 'refer_edit.png', 'tpl_edit.png']:
		p = os.path.join(output_dir, name)
		if os.path.exists(p):
			files.append(p)
	return files


def run_preprocess(video_path: str, image_path: str, width: int, height: int, retarget: bool, use_flux: bool, use_example_video: bool, use_example_image: bool, custom_output_dir: str):
	try:
		if use_example_video:
			video_path = EXAMPLE_VIDEO
		if use_example_image:
			image_path = EXAMPLE_IMAGE

		if not video_path or not os.path.exists(video_path):
			return gr.update(value=''), f'错误：视频不存在: {video_path}', None, None
		if not image_path or not os.path.exists(image_path):
			return gr.update(value=''), f'错误：参考图不存在: {image_path}', None, None

		assert not use_flux or retarget, '使用 FLUX 必须同时开启 retarget_flag'

		# 输出目录
		if custom_output_dir.strip():
			out_dir = custom_output_dir.strip()
		else:
			out_dir = os.path.join('examples/wan_animate/animate', 'process_results')
		os.makedirs(out_dir, exist_ok=True)

		# 为了与命令行逻辑一致，先把引用图复制一个正式文件名（ProcessPipeline 内部会再复制）
		# 允许用户传上传临时文件路径（Gradio 上传路径）
		# 上传文件如果是临时目录，我们直接复制一份到一个 session 目录
		if not (use_example_video and video_path == EXAMPLE_VIDEO):
			if os.path.dirname(video_path) != out_dir:
				# 保留原名
				dst_vid = os.path.join(out_dir, os.path.basename(video_path))
				if video_path != dst_vid:
					shutil.copy(video_path, dst_vid)
		if not (use_example_image and image_path == EXAMPLE_IMAGE):
			if os.path.dirname(image_path) != out_dir:
				dst_img = os.path.join(out_dir, os.path.basename(image_path))
				if image_path != dst_img:
					shutil.copy(image_path, dst_img)

		pipeline = build_pipeline(replace_flag=False, use_flux=use_flux)

		resolution_area = [int(width), int(height)]
		pipeline(video_path=video_path,
				 refer_image_path=image_path,
				 output_path=out_dir,
				 resolution_area=resolution_area,
				 fps=30,
				 iterations=3,
				 k=7,
				 w_len=1,
				 h_len=1,
				 retarget_flag=retarget,
				 use_flux=use_flux,
				 replace_flag=False)

		files = list_outputs(out_dir)
		file_list_md = '\n'.join([f'- {os.path.relpath(f)}' for f in files]) or '（未找到输出文件，请检查日志）'
		status = f'预处理完成 ✅\n输出目录: {out_dir}\n生成文件:\n{file_list_md}'

		# 关键输出展示（优先显示 src_pose / src_face / src_ref）
		pose = next((f for f in files if f.endswith('src_pose.mp4')), None)
		face = next((f for f in files if f.endswith('src_face.mp4')), None)
		ref = next((f for f in files if f.endswith('src_ref.png')), None)
		return out_dir, status, pose, face, ref
	except Exception as e:
		return gr.update(value=''), f'预处理失败: {e}\n``"""\n{traceback.format_exc()}\n"""', None, None, None


def build_ui():
    with gr.Blocks(title='Wan2.2 Animate 两种模式') as demo:
        gr.Markdown('# Wan2.2 Animate WebUI')
        gr.Markdown('提供两种模式：1) 动画模式  2) 替换模式 (replace_flag)\n均分两步：预处理 + 生成。')

        with gr.Tabs():
            # 常规模式 Tab
            with gr.Tab('动画模式'):
                gr.Markdown('### 动画模式：支持 retarget / flux')
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('#### 输入源文件')
                        video_input = gr.Video(label='驱动视频 (mp4)', sources=["upload"], interactive=True)
                        image_input = gr.Image(label='参考图 (jpg/png)', type='filepath', sources=["upload"], interactive=True)
                    with gr.Column():
                        gr.Markdown('#### 配置参数')
                        width = gr.Number(value=1280, label='resolution_area 宽')
                        height = gr.Number(value=720, label='resolution_area 高')
                        retarget_flag = gr.Checkbox(value=True, label='retarget_flag')
                        use_flux = gr.Checkbox(value=True, label='use_flux (需同时开启 retarget)')
                        custom_out = gr.Textbox(value='examples/wan_animate/animate/process_results', label='输出目录 (为空则使用默认)')
                        run_btn = gr.Button('开始预处理', variant='primary')
                status_md = gr.Markdown()
                out_dir_tb = gr.Textbox(label='输出目录', interactive=False)

                gr.Markdown('#### 关键输出预览')
                with gr.Row():
                    pose_vid = gr.Video(label='src_pose.mp4')
                    face_vid = gr.Video(label='src_face.mp4')
                    ref_img = gr.Image(label='src_ref.png')

                def _run(video_file, image_file, w, h, retarget, flux, outdir):
                    use_example_video = (video_file == EXAMPLE_VIDEO)
                    use_example_image = (image_file == EXAMPLE_IMAGE)
                    from app import run_preprocess  # local import to reuse existing function
                    return run_preprocess(
                        video_path=video_file if video_file else EXAMPLE_VIDEO,
                        image_path=image_file if image_file else EXAMPLE_IMAGE,
                        width=int(w),
                        height=int(h),
                        retarget=bool(retarget),
                        use_flux=bool(flux),
                        use_example_video=use_example_video,
                        use_example_image=use_example_image,
                        custom_output_dir=outdir or ''
                    )

                run_btn.click(
                    _run,
                    inputs=[video_input, image_input, width, height, retarget_flag, use_flux, custom_out],
                    outputs=[out_dir_tb, status_md, pose_vid, face_vid, ref_img]
                )

                gr.Markdown('#### 示例 (点击自动填充)')
                example_rows = []
                if os.path.exists(EXAMPLE_VIDEO) and os.path.exists(EXAMPLE_IMAGE):
                    example_rows.append([EXAMPLE_VIDEO, EXAMPLE_IMAGE, 1280, 720, True, True, 'examples/wan_animate/animate/process_results'])
                gr.Examples(
                    examples=example_rows,
                    inputs=[video_input, image_input, width, height, retarget_flag, use_flux, custom_out],
                    label='官方示例' if example_rows else '（未找到示例资源）'
                )

                gr.Markdown('---')
                gr.Markdown('### 第二步：生成视频 (使用上方预处理输出目录)')
                with gr.Row():
                    with gr.Column():
                        gen_refert = gr.Radio([1, 5], value=1, label='refert_num (1 或 5)')
                        gen_clip_len = gr.Slider(9, 161, value=77, step=4, label='clip_len (4n+1)')
                        gen_steps = gr.Slider(5, 50, value=20, step=1, label='sampling_steps')
                        gen_guide = gr.Slider(1.0, 6.0, value=1.0, step=0.1, label='guide_scale')
                        gen_shift = gr.Slider(0.0, 10.0, value=5.0, step=0.5, label='shift')
                        gen_solver = gr.Dropdown(['unipc', 'dpm++'], value='unipc', label='sample_solver')
                        seed_box = gr.Number(value=42, label='seed (负数随机)')
                        gen_btn = gr.Button('执行生成', variant='primary')
                    with gr.Column():
                        gen_video = gr.Video(label='生成结果视频')
                        gen_status = gr.Markdown()

                def _generate_video(process_dir, refert_num, clip_len, steps, guide, shift_v, solver, seed):
                    try:
                        if not process_dir or not os.path.isdir(process_dir):
                            return None, f'错误：无效目录 {process_dir}'
                        req_files = ['src_pose.mp4', 'src_face.mp4', 'src_ref.png']
                        missing = [f for f in req_files if not os.path.exists(os.path.join(process_dir, f))]
                        if missing:
                            return None, '缺少必要文件: ' + ', '.join(missing)
                        clip_len = int(clip_len)
                        if (clip_len - 1) % 4 != 0:
                            clip_len = ((clip_len - 1) // 4) * 4 + 1
                        model = get_animate_model('cuda:0')
                        if seed is None:
                            seed = -1
                        seed = int(seed)
                        video_tensor = model.generate(
                            src_root_path=process_dir,
                            replace_flag=False,
                            refert_num=int(refert_num),
                            clip_len=clip_len,
                            shift=float(shift_v),
                            sample_solver=str(solver),
                            sampling_steps=int(steps),
                            guide_scale=float(guide),
                            seed=seed,
                            offload_model=True,
                        )
                        cfg = WAN_CONFIGS['animate-14B']
                        out_name = f"animate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        out_path = os.path.join(process_dir, out_name)
                        save_video(tensor=video_tensor[None], save_file=out_path, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
                        return out_path, f'生成完成 ✅\n输出: {out_path}'
                    except Exception as e:
                        return None, f'生成失败: {e}\n``\n{traceback.format_exc()}\n``'

                gen_btn.click(
                    _generate_video,
                    inputs=[out_dir_tb, gen_refert, gen_clip_len, gen_steps, gen_guide, gen_shift, gen_solver, seed_box],
                    outputs=[gen_video, gen_status]
                )

                gr.Markdown('---')
                gr.Markdown('完成：动画模式 预处理 + 生成。')

            # 替换模式 Tab
            with gr.Tab('替换模式'):
                gr.Markdown('### 替换模式：使用 replace_flag，可选 use_relighting_lora')
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('#### 输入源文件 (替换)')
                        repl_video_input = gr.Video(label='驱动视频 (mp4)', sources=["upload"], interactive=True)
                        repl_image_input = gr.Image(label='参考图 (jpg/png)', type='filepath', sources=["upload"], interactive=True)
                    with gr.Column():
                        gr.Markdown('#### 预处理参数')
                        repl_width = gr.Number(value=1280, label='resolution_area 宽')
                        repl_height = gr.Number(value=720, label='resolution_area 高')
                        repl_out = gr.Textbox(value='examples/wan_animate/replace/process_results', label='输出目录 (为空则默认)')
                        repl_run_btn = gr.Button('开始预处理 (替换)', variant='primary')
                repl_status_md = gr.Markdown()
                repl_out_dir_tb = gr.Textbox(label='输出目录', interactive=False)

                gr.Markdown('#### 关键输出预览 (含 refer_edit / tpl_edit)')
                with gr.Row():
                    repl_pose_vid = gr.Video(label='src_pose.mp4')
                    repl_face_vid = gr.Video(label='src_face.mp4')
                    repl_ref_img = gr.Image(label='src_ref.png')

                def run_preprocess_replace(video_path: str, image_path: str, width: int, height: int, custom_output_dir: str):
                    try:
                        if not video_path:
                            video_path = REPLACE_EXAMPLE_VIDEO
                        if not image_path:
                            image_path = REPLACE_EXAMPLE_IMAGE
                        use_example_video = (video_path == REPLACE_EXAMPLE_VIDEO)
                        use_example_image = (image_path == REPLACE_EXAMPLE_IMAGE)
                        if not os.path.exists(video_path):
                            return gr.update(value=''), f'错误：视频不存在: {video_path}', None, None, None
                        if not os.path.exists(image_path):
                            return gr.update(value=''), f'错误：参考图不存在: {image_path}', None, None, None
                        if custom_output_dir.strip():
                            out_dir = custom_output_dir.strip()
                        else:
                            out_dir = 'examples/wan_animate/replace/process_results'
                        os.makedirs(out_dir, exist_ok=True)
                        if not (use_example_video and video_path == REPLACE_EXAMPLE_VIDEO) and os.path.dirname(video_path) != out_dir:
                            dst_vid = os.path.join(out_dir, os.path.basename(video_path))
                            if video_path != dst_vid:
                                shutil.copy(video_path, dst_vid)
                        if not (use_example_image and image_path == REPLACE_EXAMPLE_IMAGE) and os.path.dirname(image_path) != out_dir:
                            dst_img = os.path.join(out_dir, os.path.basename(image_path))
                            if image_path != dst_img:
                                shutil.copy(image_path, dst_img)
                        from app import build_pipeline, list_outputs
                        pipeline = build_pipeline(replace_flag=True, use_flux=False)
                        resolution_area = [int(width), int(height)]
                        pipeline(video_path=video_path,
                                 refer_image_path=image_path,
                                 output_path=out_dir,
                                 resolution_area=resolution_area,
                                 fps=30,
                                 iterations=3,
                                 k=7,
                                 w_len=1,
                                 h_len=1,
                                 retarget_flag=False,
                                 use_flux=False,
                                 replace_flag=True)
                        files = list_outputs(out_dir)
                        file_list_md = '\n'.join([f'- {os.path.relpath(f)}' for f in files]) or '（未找到输出文件）'
                        pose = next((f for f in files if f.endswith('src_pose.mp4')), None)
                        face = next((f for f in files if f.endswith('src_face.mp4')), None)
                        ref = next((f for f in files if f.endswith('src_ref.png')), None)
                        status = f'替换模式预处理完成 ✅\n输出目录: {out_dir}\n生成文件:\n{file_list_md}'
                        return out_dir, status, pose, face, ref
                    except Exception as e:
                        return gr.update(value=''), f'预处理失败: {e}\n``\n{traceback.format_exc()}\n``', None, None, None

                repl_run_btn.click(
                    lambda v, i, w, h, o: run_preprocess_replace(v, i, int(w), int(h), o),
                    inputs=[repl_video_input, repl_image_input, repl_width, repl_height, repl_out],
                    outputs=[repl_out_dir_tb, repl_status_md, repl_pose_vid, repl_face_vid, repl_ref_img]
                )

                gr.Markdown('#### 示例 (替换)')
                repl_example_rows = []
                if os.path.exists(REPLACE_EXAMPLE_VIDEO) and os.path.exists(REPLACE_EXAMPLE_IMAGE):
                    repl_example_rows.append([REPLACE_EXAMPLE_VIDEO, REPLACE_EXAMPLE_IMAGE, 1280, 720, 'examples/wan_animate/replace/process_results'])
                gr.Examples(
                    examples=repl_example_rows,
                    inputs=[repl_video_input, repl_image_input, repl_width, repl_height, repl_out],
                    label='替换模式示例' if repl_example_rows else '（未找到替换示例资源）'
                )

                gr.Markdown('---')
                gr.Markdown('### 第二步：生成视频 (替换模式)')
                with gr.Row():
                    with gr.Column():
                        repl_gen_refert = gr.Radio([1, 5], value=1, label='refert_num')
                        repl_use_relighting = gr.Checkbox(value=True, label='use_relighting_lora')
                        repl_seed_box = gr.Number(value=42, label='seed (负数随机)')
                        with gr.Accordion('高级参数', open=False):
                            repl_gen_clip_len = gr.Slider(9, 161, value=77, step=4, label='clip_len (4n+1)')
                            repl_gen_steps = gr.Slider(5, 50, value=20, step=1, label='sampling_steps')
                            repl_gen_guide = gr.Slider(1.0, 6.0, value=1.0, step=0.1, label='guide_scale')
                            repl_gen_shift = gr.Slider(0.0, 10.0, value=5.0, step=0.5, label='shift')
                            repl_gen_solver = gr.Dropdown(['unipc', 'dpm++'], value='unipc', label='sample_solver')
                            repl_gen_btn = gr.Button('执行生成 (替换)', variant='primary')
                    with gr.Column():
                        repl_gen_video = gr.Video(label='生成结果视频 (替换)')
                        repl_gen_status = gr.Markdown()

                def _generate_video_replace(process_dir, refert_num, use_relighting, seed, clip_len, steps, guide, shift_v, solver):
                    try:
                        if not process_dir or not os.path.isdir(process_dir):
                            return None, f'错误：无效目录 {process_dir}'
                        req_files = ['src_pose.mp4', 'src_face.mp4', 'src_ref.png']
                        missing = [f for f in req_files if not os.path.exists(os.path.join(process_dir, f))]
                        if missing:
                            return None, '缺少必要文件: ' + ', '.join(missing)
                        clip_len = int(clip_len)
                        if (clip_len - 1) % 4 != 0:
                            clip_len = ((clip_len - 1) // 4) * 4 + 1
                        model = get_animate_model('cuda:0', use_relighting_lora=bool(use_relighting))
                        if seed is None:
                            seed = -1
                        seed = int(seed)
                        video_tensor = model.generate(
                            src_root_path=process_dir,
                            replace_flag=True,
                            refert_num=int(refert_num),
                            clip_len=clip_len,
                            shift=float(shift_v),
                            sample_solver=str(solver),
                            sampling_steps=int(steps),
                            guide_scale=float(guide),
                            seed=seed,
                            offload_model=True,
                        )
                        cfg = WAN_CONFIGS['animate-14B']
                        out_name = f"animate_replace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        out_path = os.path.join(process_dir, out_name)
                        save_video(tensor=video_tensor[None], save_file=out_path, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
                        return out_path, f'替换模式生成完成 ✅\n输出: {out_path}'
                    except Exception as e:
                        return None, f'生成失败: {e}\n``\n{traceback.format_exc()}\n``'

                repl_gen_btn.click(
                    _generate_video_replace,
                    inputs=[repl_out_dir_tb, repl_gen_refert, repl_use_relighting, repl_seed_box, repl_gen_clip_len, repl_gen_steps, repl_gen_guide, repl_gen_shift, repl_gen_solver],
                    outputs=[repl_gen_video, repl_gen_status]
                )

                gr.Markdown('---')
                gr.Markdown('完成：替换模式 预处理 + 生成。')

        gr.Markdown('---')
        gr.Markdown('界面加载完成。')
    return demo


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--server_port', type=int, default=7860)
	parser.add_argument('--server_host', default='0.0.0.0')
	parser.add_argument('--share', action='store_true')
	args = parser.parse_args()
	demo = build_ui()
	demo.queue().launch(server_name=args.server_host, server_port=args.server_port, share=args.share)


if __name__ == '__main__':
	main()

