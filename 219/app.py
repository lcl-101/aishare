import ast
import os
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent
SHOT_SCRIPT = ROOT_DIR / "test_svi.py"
TALK_SCRIPT = ROOT_DIR / "test_svi_talk.py"
DANCE_SCRIPT = ROOT_DIR / "test_svi_dance.py"


def _safe_read_text(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception:
        return None


def _clean_prompt_line(text: str) -> str:
    stripped = text.strip()
    return stripped.lstrip("-•·*0123456789. \t").strip()


def _prompt_preview_text(prompt_path: Optional[str]) -> str:
    content = _safe_read_text(prompt_path)
    if not content:
        return ""

    prompts: List[str] = []
    try:
        module = ast.parse(content, mode="exec")
    except SyntaxError:
        module = None

    if module:
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.lower() in {"prompts", "prompt_list"}:
                        try:
                            value = ast.literal_eval(node.value)
                        except Exception:
                            continue
                        if isinstance(value, (list, tuple)):
                            prompts.extend(str(item) for item in value)
                        elif isinstance(value, dict):
                            prompts.extend(str(item) for item in value.values())

    if prompts:
        cleaned = [_clean_prompt_line(item) for item in prompts if isinstance(item, str)]
        non_empty = [line for line in cleaned if line]
        if non_empty:
            return "\n\n".join(non_empty)

    lines = [_clean_prompt_line(line) for line in content.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n\n".join(non_empty_lines)


def _ensure_path(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def _resolve_file_input(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, "path") and isinstance(getattr(value, "path", None), str):
        return getattr(value, "path")
    if hasattr(value, "name") and isinstance(getattr(value, "name", None), str):
        return getattr(value, "name")
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        for nested_key in ("video", "audio", "image", "file", "data"):
            nested = value.get(nested_key)
            resolved = _resolve_file_input(nested)
            if resolved:
                return resolved
    if isinstance(value, (list, tuple)):
        for item in value:
            resolved = _resolve_file_input(item)
            if resolved:
                return resolved
    return None


def _latest_media_file(directory: str, extensions: Sequence[str] = (".mp4", ".mov", ".webm", ".gif")) -> Optional[str]:
    if not directory:
        return None
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        return None
    candidates = [
        path for path in target_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _run_cli_script(script_path: Path, argv: Sequence[str]) -> None:
    original_argv = sys.argv[:]
    original_cwd = Path.cwd()
    try:
        sys.argv = [str(script_path)] + list(argv)
        os.chdir(script_path.parent)
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)


def _ensure_multitalk_weight(dit_root: str) -> None:
    target_dir = Path(dit_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "multitalk.safetensors"
    if target_path.exists():
        return

    fallback_candidates = [
        Path("weights/MeiGen-MultiTalk/multitalk.safetensors"),
        Path("weights/Stable-Video-Infinity/multitalk.safetensors"),
        Path("weights/multitalk.safetensors"),
    ]

    for fallback in fallback_candidates:
        if fallback.exists():
            try:
                relative = os.path.relpath(fallback, target_path.parent)
                target_path.symlink_to(relative)
            except FileExistsError:
                return
            except OSError:
                shutil.copy2(fallback, target_path)
            return

    raise FileNotFoundError(
        "未找到 multitalk.safetensors。请先按照 README 中的“Download Multitalk Cross-Attention”步骤下载并链接该权重。"
    )


def run_shot(
    ref_image_path: str,
    prompt_path: str,
    dit_root: str,
    extra_module_root: str,
    cfg_scale_text: float,
    num_motion_frames: int,
    num_clips: int,
    ref_pad_num: int,
    use_first_prompt_only: bool,
    output_dir: str,
) -> Optional[str]:
    cli_args: List[str] = [
        "--ref_image_path",
        ref_image_path,
        "--prompt_path",
        prompt_path,
        "--dit_root",
        dit_root,
        "--extra_module_root",
        extra_module_root,
        "--cfg_scale_text",
        str(cfg_scale_text),
        "--num_motion_frames",
        str(int(num_motion_frames)),
        "--num_clips",
        str(int(num_clips)),
        "--ref_pad_num",
        str(int(ref_pad_num)),
        "--output",
        output_dir,
    ]
    if use_first_prompt_only:
        cli_args.append("--use_first_prompt_only")
    _run_cli_script(SHOT_SCRIPT, cli_args)
    return _latest_media_file(output_dir)


def run_film(
    ref_image_path: str,
    prompt_path: str,
    dit_root: str,
    extra_module_root: str,
    cfg_scale_text: float,
    num_motion_frames: int,
    num_clips: int,
    output_dir: str,
) -> Optional[str]:
    cli_args: List[str] = [
        "--ref_image_path",
        ref_image_path,
        "--prompt_path",
        prompt_path,
        "--dit_root",
        dit_root,
        "--extra_module_root",
        extra_module_root,
        "--cfg_scale_text",
        str(cfg_scale_text),
        "--num_motion_frames",
        str(int(num_motion_frames)),
        "--num_clips",
        str(int(num_clips)),
        "--output",
        output_dir,
    ]
    _run_cli_script(SHOT_SCRIPT, cli_args)
    return _latest_media_file(output_dir)


def run_talk(
    ref_image_path: str,
    audio_path: str,
    dit_root: str,
    extra_module_root: str,
    cfg_scale_audio: float,
    cfg_scale_text: float,
    num_clips: int,
    num_steps: int,
    num_motion_frames: int,
    output_dir: str,
) -> Optional[str]:
    _ensure_multitalk_weight(dit_root)
    cli_args: List[str] = [
        "--ref_image_path",
        ref_image_path,
        "--audio_path",
        audio_path,
        "--dit_root",
        dit_root,
        "--extra_module_root",
        extra_module_root,
        "--cfg_scale_audio",
        str(cfg_scale_audio),
        "--cfg_scale_text",
        str(cfg_scale_text),
        "--num_clips",
        str(int(num_clips)),
        "--num_steps",
        str(int(num_steps)),
        "--num_motion_frames",
        str(int(num_motion_frames)),
        "--output",
        output_dir,
    ]
    _run_cli_script(TALK_SCRIPT, cli_args)
    return _latest_media_file(output_dir)


def run_dance(
    image_path: str,
    pose_path: str,
    dit_root: str,
    extra_module_root: str,
    cfg_scale_audio: float,
    cfg_scale_text: float,
    num_clips: int,
    num_steps: int,
    num_motion_frames: int,
    remove_pose: bool,
    output_dir: str,
) -> Optional[str]:
    cli_args: List[str] = [
        "--image_path",
        image_path,
        "--pose_path",
        pose_path,
        "--dit_root",
        dit_root,
        "--extra_module_root",
        extra_module_root,
        "--cfg_scale_audio",
        str(cfg_scale_audio),
        "--cfg_scale_text",
        str(cfg_scale_text),
        "--num_clips",
        str(int(num_clips)),
        "--num_steps",
        str(int(num_steps)),
        "--num_motion_frames",
        str(int(num_motion_frames)),
        "--output",
        output_dir,
    ]
    if remove_pose:
        cli_args.append("--remove_pose")
    _run_cli_script(DANCE_SCRIPT, cli_args)
    return _latest_media_file(output_dir)


def run_tom(
    ref_image_path: str,
    prompt_path: str,
    dit_root: str,
    extra_module_root: str,
    cfg_scale_text: float,
    num_motion_frames: int,
    num_clips: int,
    output_dir: str,
) -> Optional[str]:
    cli_args: List[str] = [
        "--ref_image_path",
        ref_image_path,
        "--prompt_path",
        prompt_path,
        "--dit_root",
        dit_root,
        "--extra_module_root",
        extra_module_root,
        "--cfg_scale_text",
        str(cfg_scale_text),
        "--num_motion_frames",
        str(int(num_motion_frames)),
        "--num_clips",
        str(int(num_clips)),
        "--output",
        output_dir,
    ]
    _run_cli_script(SHOT_SCRIPT, cli_args)
    return _latest_media_file(output_dir)
def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Monochrome(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate")) as demo:
        gr.HTML(
            """
        <div style='margin-bottom:8px;'>
            <h1 style='margin:0;font-size:22px;color:#0f172a;'>Stable-Video-Infinity 演示程序</h1>
            <p style='margin:4px 0 0;font-size:13px;color:#475569;'>选择不同标签，加载示例或上传素材，一键生成视频。</p>
        </div>
        """
        )

        with gr.Tabs():
            with gr.TabItem("镜头"):
                shot_example_state = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**输入设置**")
                        shot_ref_image = gr.Image(label="参考图片 (jpg/png)", type="filepath")
                        shot_prompt_file = gr.File(label="Prompt 文件 (txt)")
                        shot_prompt_preview = gr.Textbox(label="Prompt 预览", interactive=False, lines=6)
                        shot_dit_root = gr.Textbox(value="weights/Wan2.1-I2V-14B-480P/", label="dit_root", lines=1)
                        shot_extra_module = gr.Textbox(value="weights/Stable-Video-Infinity/version-1.0/svi-shot.safetensors", label="extra_module_root", lines=1)
                        with gr.Row():
                            shot_cfg = gr.Number(value=5.0, label="文本CFG", precision=2)
                            shot_num_motion = gr.Number(value=1, label="运动帧数", precision=0)
                        with gr.Row():
                            shot_num_clips = gr.Number(value=1, label="片段数", precision=0)
                            shot_ref_pad = gr.Number(value=-1, label="参考填充", precision=0)
                        shot_use_first = gr.Checkbox(value=False, label="仅用首条Prompt")
                        shot_out_dir = gr.Textbox(value="videos/svi_shot/", label="输出目录", lines=1)
                        with gr.Row():
                            shot_run_btn = gr.Button("🚀 开始生成", variant="primary")
                            shot_example_btn = gr.Button("📁 加载示例")
                        shot_status = gr.Textbox(label="状态", interactive=False, lines=2)
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**结果预览**")
                        shot_video = gr.Video(label="生成视频", interactive=False)

                def _load_shot_example():
                    ref = "data/toy_test/shot/frame.jpg"
                    prompt = "data/toy_test/shot/prompt.txt"
                    example_state = {"ref": ref, "prompt": prompt}
                    prompt_text = _prompt_preview_text(prompt) or "（示例 prompt 文件为空）"
                    status_msg = "已加载示例素材，可直接生成或重新上传。"
                    return (
                        example_state,
                        gr.update(value=ref),
                        gr.update(value=None),
                        gr.update(value=prompt_text),
                        status_msg,
                    )

                shot_example_btn.click(
                    _load_shot_example,
                    inputs=None,
                    outputs=[shot_example_state, shot_ref_image, shot_prompt_file, shot_prompt_preview, shot_status],
                )

                def _run_shot_ui(
                    ref_image_value,
                    prompt_file,
                    dit_root,
                    extra_module,
                    cfg_value,
                    num_motion_value,
                    num_clips_value,
                    ref_pad_value,
                    use_first_toggle,
                    output_dir,
                    example_state,
                ):
                    example_state = example_state or {}
                    ref_path = _resolve_file_input(ref_image_value) or example_state.get("ref")
                    prompt_path = _resolve_file_input(prompt_file) or example_state.get("prompt")
                    if not ref_path or not prompt_path:
                        return None, "请上传素材或先加载示例。", gr.update(value=None), gr.update(value="")
                    try:
                        latest = run_shot(
                            ref_path,
                            prompt_path,
                            dit_root,
                            extra_module,
                            cfg_value,
                            num_motion_value,
                            num_clips_value,
                            ref_pad_value,
                            use_first_toggle,
                            output_dir,
                        )
                        prompt_preview = _prompt_preview_text(prompt_path)
                        return (
                            latest,
                            f"完成。最新视频: {latest}",
                            gr.update(value=_ensure_path(ref_path)),
                            gr.update(value=prompt_preview),
                        )
                    except Exception as exc:  # noqa: BLE001
                        prompt_preview = _prompt_preview_text(prompt_path)
                        return None, f"错误: {exc}", gr.update(value=_ensure_path(ref_path)), gr.update(value=prompt_preview)

                shot_run_btn.click(
                    _run_shot_ui,
                    inputs=[
                        shot_ref_image,
                        shot_prompt_file,
                        shot_dit_root,
                        shot_extra_module,
                        shot_cfg,
                        shot_num_motion,
                        shot_num_clips,
                        shot_ref_pad,
                        shot_use_first,
                        shot_out_dir,
                        shot_example_state,
                    ],
                    outputs=[shot_video, shot_status, shot_ref_image, shot_prompt_preview],
                )

            with gr.TabItem("长镜头"):
                film_example_state = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**输入设置**")
                        film_ref_image = gr.Image(label="参考图片 (jpg/png)", type="filepath")
                        film_prompt_file = gr.File(label="Prompt 文件 (txt)")
                        film_prompt_preview = gr.Textbox(label="Prompt 预览", interactive=False, lines=6)
                        film_dit_root = gr.Textbox(value="weights/Wan2.1-I2V-14B-480P/", label="dit_root", lines=1)
                        film_extra_module = gr.Textbox(value="weights/Stable-Video-Infinity/version-1.0/svi-film-opt-10212025.safetensors", label="extra_module_root", lines=1)
                        with gr.Row():
                            film_cfg = gr.Number(value=5.0, label="文本CFG", precision=2)
                            film_num_motion = gr.Number(value=5, label="运动帧数", precision=0)
                        with gr.Row():
                            film_num_clips = gr.Number(value=1, label="片段数", precision=0)
                            film_out_dir = gr.Textbox(value="videos/svi_film/", label="输出目录", lines=1)
                        with gr.Row():
                            film_run_btn = gr.Button("🚀 开始生成", variant="primary")
                            film_example_btn = gr.Button("📁 加载示例")
                        film_status = gr.Textbox(label="状态", interactive=False, lines=2)
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**结果预览**")
                        film_video = gr.Video(label="生成视频", interactive=False)

                def _load_film_example():
                    ref = "data/toy_test/film/frame.jpg"
                    prompt = "data/toy_test/film/prompt.txt"
                    example_state = {"ref": ref, "prompt": prompt}
                    prompt_text = _prompt_preview_text(prompt) or "（示例 prompt 文件为空）"
                    status_msg = "已加载示例素材，可直接生成或重新上传。"
                    return (
                        example_state,
                        gr.update(value=ref),
                        gr.update(value=None),
                        gr.update(value=prompt_text),
                        status_msg,
                    )

                film_example_btn.click(
                    _load_film_example,
                    inputs=None,
                    outputs=[film_example_state, film_ref_image, film_prompt_file, film_prompt_preview, film_status],
                )

                def _run_film_ui(
                    ref_image_value,
                    prompt_file,
                    dit_root,
                    extra_module,
                    cfg_value,
                    num_motion_value,
                    num_clips_value,
                    output_dir,
                    example_state,
                ):
                    example_state = example_state or {}
                    ref_path = _resolve_file_input(ref_image_value) or example_state.get("ref")
                    prompt_path = _resolve_file_input(prompt_file) or example_state.get("prompt")
                    if not ref_path or not prompt_path:
                        return None, "请上传素材或先加载示例。", gr.update(value=None), gr.update(value="")
                    try:
                        latest = run_film(
                            ref_path,
                            prompt_path,
                            dit_root,
                            extra_module,
                            cfg_value,
                            num_motion_value,
                            num_clips_value,
                            output_dir,
                        )
                        prompt_preview = _prompt_preview_text(prompt_path)
                        return (
                            latest,
                            f"完成。最新视频: {latest}",
                            gr.update(value=_ensure_path(ref_path)),
                            gr.update(value=prompt_preview),
                        )
                    except Exception as exc:  # noqa: BLE001
                        prompt_preview = _prompt_preview_text(prompt_path)
                        return None, f"错误: {exc}", gr.update(value=_ensure_path(ref_path)), gr.update(value=prompt_preview)

                film_run_btn.click(
                    _run_film_ui,
                    inputs=[
                        film_ref_image,
                        film_prompt_file,
                        film_dit_root,
                        film_extra_module,
                        film_cfg,
                        film_num_motion,
                        film_num_clips,
                        film_out_dir,
                        film_example_state,
                    ],
                    outputs=[film_video, film_status, film_ref_image, film_prompt_preview],
                )

            with gr.TabItem("对话"):
                talk_example_state = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**输入设置**")
                        talk_ref_image = gr.Image(label="参考图片 (png)", type="filepath")
                        talk_audio = gr.Audio(label="音频文件 (wav)", type="filepath", sources=["upload"])
                        talk_dit_root = gr.Textbox(value="weights/Wan2.1-I2V-14B-480P/", label="dit_root", lines=1)
                        talk_extra_module = gr.Textbox(value="weights/Stable-Video-Infinity/version-1.0/svi-talk.safetensors", label="extra_module_root", lines=1)
                        with gr.Row():
                            talk_cfg_audio = gr.Number(value=5.0, label="音频CFG", precision=2)
                            talk_cfg_text = gr.Number(value=2.0, label="文本CFG", precision=2)
                        with gr.Row():
                            talk_num_clips = gr.Number(value=1, label="片段数", precision=0)
                            talk_num_steps = gr.Number(value=50, label="步骤数", precision=0)
                        talk_num_motion = gr.Number(value=1, label="运动帧数", precision=0)
                        talk_out_dir = gr.Textbox(value="videos/svi_talk/", label="输出目录", lines=1)
                        with gr.Row():
                            talk_run_btn = gr.Button("🎙️ 开始生成", variant="primary")
                            talk_example_btn = gr.Button("📁 加载示例")
                        talk_status = gr.Textbox(label="状态", interactive=False, lines=2)
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**结果预览**")
                        talk_video = gr.Video(label="生成视频", interactive=False)

                def _load_talk_example():
                    ref = "data/toy_test/talk/obama.png"
                    audio = "data/toy_test/talk/obama_5min.wav"
                    example_state = {"ref": ref, "audio": audio}
                    status_msg = "已加载示例素材，可直接生成或重新上传。"
                    return (
                        example_state,
                        gr.update(value=ref),
                        gr.update(value=audio),
                        status_msg,
                    )

                talk_example_btn.click(
                    _load_talk_example,
                    inputs=None,
                    outputs=[talk_example_state, talk_ref_image, talk_audio, talk_status],
                )

                def _run_talk_ui(
                    ref_image_value,
                    audio_value,
                    dit_root,
                    extra_module,
                    cfg_audio_value,
                    cfg_text_value,
                    num_clips_value,
                    num_steps_value,
                    num_motion_value,
                    output_dir,
                    example_state,
                ):
                    example_state = example_state or {}
                    ref_path = _resolve_file_input(ref_image_value) or example_state.get("ref")
                    audio_path = _resolve_file_input(audio_value) or example_state.get("audio")
                    if not ref_path or not audio_path:
                        return None, "请上传素材或先加载示例。", gr.update(value=None), gr.update(value=None)
                    try:
                        latest = run_talk(
                            ref_path,
                            audio_path,
                            dit_root,
                            extra_module,
                            cfg_audio_value,
                            cfg_text_value,
                            num_clips_value,
                            num_steps_value,
                            num_motion_value,
                            output_dir,
                        )
                        return (
                            latest,
                            f"完成。最新视频: {latest}",
                            gr.update(value=_ensure_path(ref_path)),
                            gr.update(value=_ensure_path(audio_path)),
                        )
                    except Exception as exc:  # noqa: BLE001
                        return (
                            None,
                            f"错误: {exc}",
                            gr.update(value=_ensure_path(ref_path)),
                            gr.update(value=_ensure_path(audio_path)),
                        )

                talk_run_btn.click(
                    _run_talk_ui,
                    inputs=[
                        talk_ref_image,
                        talk_audio,
                        talk_dit_root,
                        talk_extra_module,
                        talk_cfg_audio,
                        talk_cfg_text,
                        talk_num_clips,
                        talk_num_steps,
                        talk_num_motion,
                        talk_out_dir,
                        talk_example_state,
                    ],
                    outputs=[talk_video, talk_status, talk_ref_image, talk_audio],
                )

            with gr.TabItem("舞蹈"):
                dance_example_state = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**输入设置**")
                        dance_image = gr.Image(label="参考图片 (png)", type="filepath")
                        dance_pose = gr.Video(label="姿态视频 (mp4)", sources=["upload"], interactive=True)
                        dance_dit_root = gr.Textbox(value="./weights/Wan2.1-I2V-14B-480P/", label="dit_root", lines=1)
                        dance_extra_module = gr.Textbox(value="weights/Stable-Video-Infinity/version-1.0/svi-dance.safetensors", label="extra_module_root", lines=1)
                        with gr.Row():
                            dance_cfg_audio = gr.Number(value=1.0, label="音频CFG", precision=2)
                            dance_cfg_text = gr.Number(value=2.0, label="文本CFG", precision=2)
                        with gr.Row():
                            dance_num_clips = gr.Number(value=1, label="片段数", precision=0)
                            dance_num_steps = gr.Number(value=50, label="步骤数", precision=0)
                        dance_num_motion = gr.Number(value=1, label="运动帧数", precision=0)
                        dance_remove_pose = gr.Checkbox(value=False, label="移除姿态")
                        dance_out_dir = gr.Textbox(value="videos/svi_dance/", label="输出目录", lines=1)
                        with gr.Row():
                            dance_run_btn = gr.Button("💃 开始生成", variant="primary")
                            dance_example_btn = gr.Button("📁 加载示例")
                        dance_status = gr.Textbox(label="状态", interactive=False, lines=2)
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**结果预览**")
                        dance_video = gr.Video(label="生成视频", interactive=False)

                def _load_dance_example():
                    img_path = "data/toy_test/dance/image.png"
                    pose_path = "data/toy_test/dance/pose.mp4"
                    example_state = {"ref": img_path, "pose": pose_path}
                    status_msg = "已加载示例素材，可直接生成或重新上传。"
                    return (
                        example_state,
                        gr.update(value=img_path),
                        gr.update(value=pose_path),
                        status_msg,
                    )

                dance_example_btn.click(
                    _load_dance_example,
                    inputs=None,
                    outputs=[dance_example_state, dance_image, dance_pose, dance_status],
                )

                def _run_dance_ui(
                    image_value,
                    pose_value,
                    dit_root,
                    extra_module,
                    cfg_audio_value,
                    cfg_text_value,
                    num_clips_value,
                    num_steps_value,
                    num_motion_value,
                    remove_pose_toggle,
                    output_dir,
                    example_state,
                ):
                    example_state = example_state or {}
                    image_path = _resolve_file_input(image_value) or example_state.get("ref")
                    pose_path = _resolve_file_input(pose_value) or example_state.get("pose")
                    if not image_path or not pose_path:
                        return None, "请上传素材或先加载示例。", gr.update(value=None), gr.update(value=None)
                    try:
                        latest = run_dance(
                            image_path,
                            pose_path,
                            dit_root,
                            extra_module,
                            cfg_audio_value,
                            cfg_text_value,
                            num_clips_value,
                            num_steps_value,
                            num_motion_value,
                            remove_pose_toggle,
                            output_dir,
                        )
                        return (
                            latest,
                            f"完成。最新视频: {latest}",
                            gr.update(value=_ensure_path(image_path)),
                            gr.update(value=_ensure_path(pose_path)),
                        )
                    except Exception as exc:  # noqa: BLE001
                        return (
                            None,
                            f"错误: {exc}",
                            gr.update(value=_ensure_path(image_path)),
                            gr.update(value=_ensure_path(pose_path)),
                        )

                dance_run_btn.click(
                    _run_dance_ui,
                    inputs=[
                        dance_image,
                        dance_pose,
                        dance_dit_root,
                        dance_extra_module,
                        dance_cfg_audio,
                        dance_cfg_text,
                        dance_num_clips,
                        dance_num_steps,
                        dance_num_motion,
                        dance_remove_pose,
                        dance_out_dir,
                        dance_example_state,
                    ],
                    outputs=[dance_video, dance_status, dance_image, dance_pose],
                )

            with gr.TabItem("Tom"):
                tom_example_state = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**输入设置**")
                        tom_ref_image = gr.Image(label="参考图片 (png)", type="filepath")
                        tom_prompt_file = gr.File(label="Prompt 文件 (txt)")
                        tom_prompt_preview = gr.Textbox(label="Prompt 预览", interactive=False, lines=6)
                        tom_dit_root = gr.Textbox(value="weights/Wan2.1-I2V-14B-480P/", label="dit_root", lines=1)
                        tom_extra_module = gr.Textbox(value="weights/Stable-Video-Infinity/version-1.0/svi-tom.safetensors", label="extra_module_root", lines=1)
                        with gr.Row():
                            tom_cfg = gr.Number(value=5.0, label="文本CFG", precision=2)
                            tom_num_motion = gr.Number(value=1, label="运动帧数", precision=0)
                        tom_num_clips = gr.Number(value=1, label="片段数", precision=0)
                        tom_out_dir = gr.Textbox(value="videos/svi_tom/", label="输出目录", lines=1)
                        with gr.Row():
                            tom_run_btn = gr.Button("🧢 开始生成", variant="primary")
                            tom_example_btn = gr.Button("📁 加载示例")
                        tom_status = gr.Textbox(label="状态", interactive=False, lines=2)
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown("**结果预览**")
                        tom_video = gr.Video(label="生成视频", interactive=False)

                def _load_tom_example():
                    ref = "data/toy_test/tom/frame.png"
                    prompt = "data/toy_test/tom/prompt.txt"
                    example_state = {"ref": ref, "prompt": prompt}
                    prompt_text = _prompt_preview_text(prompt) or "（示例 prompt 文件为空）"
                    status_msg = "已加载示例素材，可直接生成或重新上传。"
                    return (
                        example_state,
                        gr.update(value=ref),
                        gr.update(value=None),
                        gr.update(value=prompt_text),
                        status_msg,
                    )

                tom_example_btn.click(
                    _load_tom_example,
                    inputs=None,
                    outputs=[tom_example_state, tom_ref_image, tom_prompt_file, tom_prompt_preview, tom_status],
                )

                def _run_tom_ui(
                    ref_image_value,
                    prompt_file,
                    dit_root,
                    extra_module,
                    cfg_value,
                    num_motion_value,
                    num_clips_value,
                    output_dir,
                    example_state,
                ):
                    example_state = example_state or {}
                    ref_path = _resolve_file_input(ref_image_value) or example_state.get("ref")
                    prompt_path = _resolve_file_input(prompt_file) or example_state.get("prompt")
                    if not ref_path or not prompt_path:
                        return None, "请上传素材或先加载示例。", gr.update(value=None), gr.update(value="")
                    try:
                        latest = run_tom(
                            ref_path,
                            prompt_path,
                            dit_root,
                            extra_module,
                            cfg_value,
                            num_motion_value,
                            num_clips_value,
                            output_dir,
                        )
                        prompt_preview = _prompt_preview_text(prompt_path)
                        return (
                            latest,
                            f"完成。最新视频: {latest}",
                            gr.update(value=_ensure_path(ref_path)),
                            gr.update(value=prompt_preview),
                        )
                    except Exception as exc:  # noqa: BLE001
                        prompt_preview = _prompt_preview_text(prompt_path)
                        return None, f"错误: {exc}", gr.update(value=_ensure_path(ref_path)), gr.update(value=prompt_preview)

                tom_run_btn.click(
                    _run_tom_ui,
                    inputs=[
                        tom_ref_image,
                        tom_prompt_file,
                        tom_dit_root,
                        tom_extra_module,
                        tom_cfg,
                        tom_num_motion,
                        tom_num_clips,
                        tom_out_dir,
                        tom_example_state,
                    ],
                    outputs=[tom_video, tom_status, tom_ref_image, tom_prompt_preview],
                )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", share=False)
