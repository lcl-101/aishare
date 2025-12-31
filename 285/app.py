import argparse
import codecs as cs
import json
import os
import os.path as osp
import random
import re
import textwrap
from typing import List, Optional, Tuple, Union

import torch

import gradio as gr


# 本地模型路径（已提前下载）
LOCAL_MODEL_PATH = "./checkpoints/HY-Motion-1.0/HY-Motion-1.0"
LOCAL_PROMPTER_PATH = "./checkpoints/Text2MotionPrompter"
LOCAL_QWEN_PATH = "./checkpoints/Qwen3-8B"
LOCAL_CLIP_PATH = "./checkpoints/clip-vit-large-patch14"

# 创建 ckpts 目录下的符号链接，指向 checkpoints 中的模型（兼容 text_encoder.py 中的硬编码路径）
CKPTS_DIR = "./ckpts"
CHECKPOINTS_DIR = "./checkpoints"
if os.path.isdir(CHECKPOINTS_DIR):
    os.makedirs(CKPTS_DIR, exist_ok=True)
    # 需要链接的模型目录
    models_to_link = ["Qwen3-8B", "clip-vit-large-patch14", "Text2MotionPrompter"]
    for model_name in models_to_link:
        src_path = os.path.join(CHECKPOINTS_DIR, model_name)
        dst_path = os.path.join(CKPTS_DIR, model_name)
        if os.path.isdir(src_path) and not os.path.exists(dst_path):
            try:
                os.symlink(os.path.abspath(src_path), dst_path)
                print(f">>> 已创建符号链接: {dst_path} -> {src_path}")
            except OSError as e:
                print(f">>> 创建符号链接失败 ({model_name}): {e}")

# 设置 Text2MotionPrompter 本地路径
if "PROMPT_MODEL_PATH" not in os.environ and os.path.isdir(LOCAL_PROMPTER_PATH):
    os.environ["PROMPT_MODEL_PATH"] = LOCAL_PROMPTER_PATH

# 检测本地文本编码器模型，若存在则使用本地模型（USE_HF_MODELS=0）
# 需要下载: Qwen/Qwen3-8B -> checkpoints/Qwen3-8B
#          openai/clip-vit-large-patch14 -> checkpoints/clip-vit-large-patch14
if "USE_HF_MODELS" not in os.environ:
    if os.path.isdir(LOCAL_QWEN_PATH) and os.path.isdir(LOCAL_CLIP_PATH):
        os.environ["USE_HF_MODELS"] = "0"  # 使用本地模型
        print(f">>> 检测到本地文本编码器模型，将使用本地路径加载")
    else:
        os.environ["USE_HF_MODELS"] = "1"  # 从 HuggingFace 下载
        print(f">>> 未检测到本地文本编码器模型，将从 HuggingFace 下载")
        if not os.path.isdir(LOCAL_QWEN_PATH):
            print(f">>>   缺少: {LOCAL_QWEN_PATH} (请下载 Qwen/Qwen3-8B)")
        if not os.path.isdir(LOCAL_CLIP_PATH):
            print(f">>>   缺少: {LOCAL_CLIP_PATH} (请下载 openai/clip-vit-large-patch14)")


# Import spaces for Hugging Face Zero GPU support
try:
    import spaces

    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

    # Create a dummy decorator when spaces is not available
    class spaces:
        @staticmethod
        def GPU(func=None, duration=None):
            def decorator(fn):
                return fn

            if func is not None:
                return func
            return decorator


from hymotion.utils.t2m_runtime import T2MRuntime

NUM_WORKERS = torch.cuda.device_count() if torch.cuda.is_available() else 1

# Global runtime instance for Zero GPU lazy loading
_global_runtime = None
_global_args = None


def _init_runtime_if_needed():
    """Initialize runtime lazily for Zero GPU support."""
    global _global_runtime, _global_args
    if _global_runtime is not None:
        return _global_runtime

    if _global_args is None:
        raise RuntimeError("Runtime args not set. Call set_runtime_args() first.")

    args = _global_args
    cfg = osp.join(args.model_path, "config.yml")
    ckpt = osp.join(args.model_path, "latest.ckpt")

    skip_model_loading = False
    if not os.path.exists(ckpt):
        print(f">>> [警告] 检查点文件未找到: {ckpt}")
        print(f">>> [警告] 模型加载将被跳过，动作生成功能将不可用。")
        skip_model_loading = True

    print(">>> 正在初始化 T2MRuntime...")

    skip_text = False
    _global_runtime = T2MRuntime(
        config_path=cfg,
        ckpt_name=ckpt,
        skip_text=skip_text,
        device_ids=None,
        skip_model_loading=skip_model_loading,
        disable_prompt_engineering=args.disable_prompt_engineering,
        prompt_engineering_host=args.prompt_engineering_host,
        prompt_engineering_model_path=args.prompt_engineering_model_path,
    )
    return _global_runtime


@spaces.GPU(duration=120)
def generate_motion_on_gpu(
    text: str,
    seeds_csv: str,
    motion_duration: float,
    cfg_scale: float,
    output_format: str,
    original_text: str,
    output_dir: str,
) -> Tuple[str, List[str]]:
    """
    GPU-decorated function for motion generation.
    This function will request GPU allocation on Hugging Face Zero GPU.
    """
    runtime = _init_runtime_if_needed()

    html_content, fbx_files, _ = runtime.generate_motion(
        text=text,
        seeds_csv=seeds_csv,
        duration=motion_duration,
        cfg_scale=cfg_scale,
        output_format=output_format,
        original_text=original_text,
        output_dir=output_dir,
    )
    return html_content, fbx_files


# 定义数据源
DATA_SOURCES = {
    "example_prompts": "examples/example_prompts/example_subset.json",
}

# 创建界面样式
APP_CSS = """
    :root{
    --primary-start:#667eea; --primary-end:#764ba2;
    --secondary-start:#4facfe; --secondary-end:#00f2fe;
    --accent-start:#f093fb; --accent-end:#f5576c;
    --page-bg:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
    --card-bg:linear-gradient(135deg,#ffffff 0%,#f8f9fa 100%);
    --radius:12px;
    --iframe-bg:#ffffff;
    }

    /* 深色模式变量 */
    [data-theme="dark"], .dark {
    --page-bg:linear-gradient(135deg,#1a1a1a 0%,#2d3748 100%);
    --card-bg:linear-gradient(135deg,#2d3748 0%,#374151 100%);
    --text-primary:#f7fafc;
    --text-secondary:#e2e8f0;
    --border-color:#4a5568;
    --input-bg:#374151;
    --input-border:#4a5568;
    --iframe-bg:#1a1a2e;
    }

    /* 页面和卡片 */
    .gradio-container{
    background:var(--page-bg) !important;
    min-height:100vh !important;
    color:var(--text-primary, #333) !important;
    }

    .main-header{
    background:transparent !important; border:none !important; box-shadow:none !important;
    padding:0 !important; margin:10px 0 16px !important;
    text-align:center !important;
    }

    .main-header h1, .main-header p, .main-header li {
    color:var(--text-primary, #333) !important;
    }

    .left-panel,.right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important;
    }

    .gradio-accordion{
    border:1px solid var(--border-color, #e1e5e9) !important;
    border-radius:var(--radius) !important;
    margin:12px 0 !important; background:transparent !important;
    }

    .gradio-accordion summary{
    background:transparent !important;
    padding:14px 18px !important;
    font-weight:600 !important;
    color:var(--text-primary, #495057) !important;
    }

    .gradio-group{
    background:transparent !important; border:none !important;
    border-radius:8px !important; padding:12px 0 !important; margin:8px 0 !important;
    }

    /* 输入框样式 - 深色模式适配 */
    .gradio-textbox input,.gradio-textbox textarea,.gradio-dropdown .wrap{
    border-radius:8px !important;
    border:2px solid var(--input-border, #e9ecef) !important;
    background:var(--input-bg, #fff) !important;
    color:var(--text-primary, #333) !important;
    transition:.2s all !important;
    }

    .gradio-textbox input:focus,.gradio-textbox textarea:focus,.gradio-dropdown .wrap:focus-within{
    border-color:var(--primary-start) !important;
    box-shadow:0 0 0 3px rgba(102,126,234,.1) !important;
    }

    .gradio-slider input[type="range"]{
    background:linear-gradient(to right,var(--primary-start),var(--primary-end)) !important;
    border-radius:10px !important;
    }

    .gradio-checkbox input[type="checkbox"]{
    border-radius:4px !important;
    border:2px solid var(--input-border, #e9ecef) !important;
    transition:.2s all !important;
    }

    .gradio-checkbox input[type="checkbox"]:checked{
    background:linear-gradient(45deg,var(--primary-start),var(--primary-end)) !important;
    border-color:var(--primary-start) !important;
    }

    /* 标签文字颜色适配 */
    .gradio-textbox label, .gradio-dropdown label, .gradio-slider label,
    .gradio-checkbox label, .gradio-html label {
    color:var(--text-primary, #333) !important;
    }

    .gradio-textbox .info, .gradio-dropdown .info, .gradio-slider .info,
    .gradio-checkbox .info {
    color:var(--text-secondary, #666) !important;
    }

    /* 状态信息 - 深色模式适配 */
    .gradio-textbox[data-testid*="状态信息"] input{
    background:var(--input-bg, linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%)) !important;
    border:2px solid var(--input-border, #dee2e6) !important;
    color:var(--text-primary, #495057) !important;
    font-weight:500 !important;
    }

    /* 按钮基础类和变体 */
    .generate-button,.rewrite-button,.dice-button{
    border:none !important; color:#fff !important; font-weight:600 !important;
    border-radius:8px !important; transition:.3s all !important;
    box-shadow:0 4px 15px rgba(0,0,0,.12) !important;
    }

    .generate-button{ background:linear-gradient(45deg,var(--primary-start),var(--primary-end)) !important; }
    .rewrite-button{ background:linear-gradient(45deg,var(--secondary-start),var(--secondary-end)) !important; }
    .dice-button{
    background:linear-gradient(45deg,var(--accent-start),var(--accent-end)) !important;
    height:40px !important;
    }

    .generate-button:hover,.rewrite-button:hover{ transform:translateY(-2px) !important; }
    .dice-button:hover{
    transform:scale(1.05) !important;
    box-shadow:0 4px 12px rgba(240,147,251,.28) !important;
    }

    .dice-container{
    display:flex !important;
    align-items:flex-end !important;
    justify-content:center !important;
    }

    /* 右侧面板裁剪溢出，避免双滚动条 */
    .right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important; overflow:hidden !important;
    }

    /* 主内容行 - 确保等高 */
    .main-row {
    display: flex !important;
    align-items: stretch !important;
    }

    /* Flask 显示区域 - 匹配左侧面板高度 */
    .flask-display{
    padding:0 !important; margin:0 !important; border:none !important;
    box-shadow:none !important; background:var(--iframe-bg) !important;
    border-radius:10px !important; position:relative !important;
    height:100% !important; min-height:750px !important;
    display:flex !important; flex-direction:column !important;
    }

    .flask-display iframe{
    width:100% !important; flex:1 !important; min-height:750px !important;
    border:none !important; border-radius:10px !important; display:block !important;
    background:var(--iframe-bg) !important;
    }

    /* 右侧面板应拉伸以匹配左侧面板 */
    .right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important; overflow:hidden !important;
    display:flex !important; flex-direction:column !important;
    }

    /* 确保下拉菜单在深色模式下可见 */
    [data-theme="dark"] .gradio-dropdown .wrap,
    .dark .gradio-dropdown .wrap {
    background:var(--input-bg) !important;
    color:var(--text-primary) !important;
    }

    [data-theme="dark"] .gradio-dropdown .option,
    .dark .gradio-dropdown .option {
    background:var(--input-bg) !important;
    color:var(--text-primary) !important;
    }

    [data-theme="dark"] .gradio-dropdown .option:hover,
    .dark .gradio-dropdown .option:hover {
    background:var(--border-color) !important;
    }

    .footer{
    text-align:center !important;
    margin-top:20px !important;
    padding:10px !important;
    color:var(--text-secondary, #666) !important;
    }
"""

HEADER_BASE_MD = "# HY-Motion-1.0: 文本生成动作演示平台"

FOOTER_MD = "*这是一个测试版本，欢迎反馈任何问题或建议！*"

HTML_OUTPUT_PLACEHOLDER = """
<div style='height: 750px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
    <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">欢迎使用 HY-Motion-1.0！</p>
        <p style="color: #8d8d8d;">这里还没有动作可视化内容。</p>
    </div>
</div>
"""


def load_examples_from_txt(txt_path: str, example_record_fps=20, max_duration=12):
    """从文本文件加载示例。"""

    def _parse_line(line: str) -> Optional[Tuple[str, float]]:
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split("#")
            if len(parts) >= 2:
                text = parts[0].strip()
                duration = int(parts[1]) / example_record_fps
                duration = min(duration, max_duration)
            else:
                text = line.strip()
                duration = 5.0
            return text, duration
        return None

    examples: List[Tuple[str, float]] = []
    if os.path.exists(txt_path):
        try:
            if txt_path.endswith(".txt"):
                with cs.open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        result = _parse_line(line)
                        if result is None:
                            continue
                        text, duration = result
                        examples.append((text, duration))
            elif txt_path.endswith(".json"):
                with cs.open(txt_path, "r", encoding="utf-8") as f:
                    lines = json.load(f)
                    for key, value in lines.items():
                        if "_raw_chn" in key or "GENERATE_PROMPT_FORMAT" in key:
                            continue
                        for line in value:
                            result = _parse_line(line)
                            if result is None:
                                continue
                            text, duration = result
                            examples.append((text, duration))
            print(f">>> 从 {txt_path} 加载了 {len(examples)} 个示例")
        except Exception as e:
            print(f">>> 从 {txt_path} 加载示例失败: {e}")
    else:
        print(f">>> 示例文件未找到: {txt_path}")

    return examples


class T2MGradioUI:
    def __init__(self, runtime: T2MRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args

        # 检查重写功能是否可用:
        # - 提供了 prompt_engineering_host（使用远程API）
        # - 或者本地 prompter 模型存在（使用本地模型）
        print(f">>> args: {vars(args)}")

        has_remote_host = args.prompt_engineering_host is not None and args.prompt_engineering_host.strip() != ""

        # 检查本地 prompter 模型是否存在
        local_prompter_path = "./checkpoints/Text2MotionPrompter"
        has_local_prompter = os.path.exists(local_prompter_path) and os.path.isdir(local_prompter_path)

        if has_local_prompter:
            print(f">>> 找到本地 prompter 模型: {local_prompter_path}")

        self.prompt_engineering_available = (
            has_remote_host or has_local_prompter
        ) and not args.disable_prompt_engineering

        print(
            f">>> 提示词工程可用: {self.prompt_engineering_available} (远程: {has_remote_host}, 本地: {has_local_prompter})"
        )

        self.all_example_data = {}
        self._init_example_data()

    def _init_example_data(self):
        for source_name, file_path in DATA_SOURCES.items():
            examples = load_examples_from_txt(file_path)
            if examples:
                self.all_example_data[source_name] = examples
            else:
                # 提供默认示例作为备选
                self.all_example_data[source_name] = [
                    ("Twist at the waist and punch across the body.", 3.0),
                    ("A person is running then takes big leap.", 3.0),
                    ("A person holds a railing and walks down a set of stairs.", 5.0),
                    (
                        "A man performs a fluid and rhythmic hip-hop style dance, incorporating body waves, arm gestures, and side steps.",
                        5.0,
                    ),
                ]
        print(f">>> 已加载数据源: {list(self.all_example_data.keys())}")

    def _get_header_text(self):
        return HEADER_BASE_MD

    def _generate_random_seeds(self):
        seeds = [random.randint(0, 999) for _ in range(4)]
        return ",".join(map(str, seeds))

    def _prompt_engineering(
        self, text: str, duration: float, enable_rewrite: bool = True, enable_duration_est: bool = True
    ):
        if not text.strip():
            return "", gr.update(interactive=False), gr.update()

        call_llm = enable_rewrite or enable_duration_est
        if not call_llm:
            print(f"\t>>> 使用原始时长和原始文本...")
            predicted_duration = duration
            rewritten_text = text
        else:
            print(f"\t>>> 使用 LLM 估算时长/重写文本...")
            try:
                predicted_duration, rewritten_text = self.runtime.rewrite_text_and_infer_time(text=text)
            except Exception as e:
                print(f"\t>>> 文本重写/时长预测失败: {e}")
                return (
                    f"❌ 文本重写/时长预测失败: {str(e)}",
                    gr.update(interactive=False),
                    gr.update(),
                )
            if not enable_rewrite:
                rewritten_text = text
            if not enable_duration_est:
                predicted_duration = duration

        return rewritten_text, gr.update(interactive=True), gr.update(value=predicted_duration)

    def _generate_motion(
        self,
        original_text: str,
        rewritten_text: str,
        seed_input: str,
        duration: float,
        cfg_scale: float,
    ) -> Tuple[str, List[str]]:
        # 当重写功能不可用时，直接使用原始文本
        if not self.prompt_engineering_available:
            text_to_use = original_text.strip()
            if not text_to_use:
                return "错误：输入文本为空，请先输入文本", []
        else:
            text_to_use = rewritten_text.strip()
            if not text_to_use:
                return "错误：重写文本为空，请先重写文本", []

        try:
            # 如果全局运行时可用（用于 Zero GPU），则使用它，否则使用 self.runtime
            runtime = _global_runtime if _global_runtime is not None else self.runtime
            fbx_ok = getattr(runtime, "fbx_available", False)
            req_format = "fbx" if fbx_ok else "dict"

            # 使用 GPU 装饰的函数支持 Zero GPU
            html_content, fbx_files = generate_motion_on_gpu(
                text=text_to_use,
                seeds_csv=seed_input,
                motion_duration=duration,
                cfg_scale=cfg_scale,
                output_format=req_format,
                original_text=original_text,
                output_dir=self.args.output_dir,
            )
            # 转义 HTML 内容用于 srcdoc 属性
            escaped_html = html_content.replace('"', "&quot;")
            # 返回带有 srcdoc 的 iframe - 直接嵌入 HTML 内容
            iframe_html = f"""
                <iframe
                    srcdoc="{escaped_html}"
                    width="100%"
                    height="750px"
                    style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
                ></iframe>
            """
            return iframe_html, fbx_files
        except Exception as e:
            print(f"\t>>> 动作生成失败: {e}")
            return (
                f"❌ 动作生成失败: {str(e)}\n\n请检查输入参数或稍后重试",
                [],
            )

    def _get_example_choices(self):
        """获取所有数据源的示例选项"""
        choices = ["自定义输入"]
        for source_name in self.all_example_data:
            example_data = self.all_example_data[source_name]
            for text, _ in example_data:
                display_text = f"{text[:50]}..." if len(text) > 50 else text
                choices.append(display_text)
        return choices

    def _on_example_select(self, selected_example):
        """选择示例时的回调函数"""
        if selected_example == "自定义输入":
            return "", self._generate_random_seeds(), gr.update()
        else:
            # 从所有数据源中查找对应的示例
            for source_name in self.all_example_data:
                example_data = self.all_example_data[source_name]
                for text, duration in example_data:
                    display_text = f"{text[:50]}..." if len(text) > 50 else text
                    if display_text == selected_example:
                        return text, self._generate_random_seeds(), gr.update(value=duration)
            return "", self._generate_random_seeds(), gr.update()

    def build_ui(self):
        with gr.Blocks(css=APP_CSS) as demo:
            self.header_md = gr.Markdown(HEADER_BASE_MD, elem_classes=["main-header"])

            with gr.Row():
                # 左侧控制面板
                with gr.Column(scale=2, elem_classes=["left-panel"]):
                    # 输入文本框
                    self.text_input = gr.Textbox(
                        label="📝 输入文本",
                        placeholder="输入文本以生成动作，支持中英文文本输入。",
                    )
                    # 重写文本框
                    self.rewritten_text = gr.Textbox(
                        label="✏️ 重写文本",
                        placeholder="重写后的文本将显示在这里，您可以进一步编辑",
                        interactive=True,
                        visible=False,
                    )
                    # 时长滑块
                    self.duration_slider = gr.Slider(
                        minimum=0.5,
                        maximum=12,
                        value=5.0,
                        step=0.1,
                        label="⏱️ 动作时长（秒）",
                        info="可自由调整动作时长",
                    )

                    # 执行按钮
                    with gr.Row():
                        if self.prompt_engineering_available:
                            self.rewrite_btn = gr.Button(
                                "🔄 重写文本",
                                variant="secondary",
                                size="lg",
                                elem_classes=["rewrite-button"],
                            )
                        else:
                            # 创建一个隐藏/禁用的占位按钮
                            self.rewrite_btn = gr.Button(
                                "🔄 重写文本（不可用）",
                                variant="secondary",
                                size="lg",
                                elem_classes=["rewrite-button"],
                                interactive=False,
                                visible=False,
                            )

                        self.generate_btn = gr.Button(
                            "🚀 生成动作",
                            variant="primary",
                            size="lg",
                            elem_classes=["generate-button"],
                            interactive=not self.prompt_engineering_available,  # 如果重写不可用则直接启用
                        )

                    if not self.prompt_engineering_available:
                        gr.Markdown(
                            "> ⚠️ **提示词工程不可用。** 文本重写和时长估算功能已禁用。将直接使用您输入的文本和时长。"
                        )

                    # 高级设置
                    with gr.Accordion("🔧 高级设置", open=False):
                        self._build_advanced_settings()

                    # 示例选择下拉框
                    self.example_dropdown = gr.Dropdown(
                        choices=self._get_example_choices(),
                        value="自定义输入",
                        label="📚 测试示例",
                        info="选择预设示例或在上方输入您自己的文本",
                        interactive=True,
                    )

                    # 状态消息取决于重写功能是否可用
                    if self.prompt_engineering_available:
                        status_msg = "请先点击 [🔄 重写文本] 按钮重写文本"
                    else:
                        status_msg = "输入文本后直接点击 [🚀 生成动作]。"

                    self.status_output = gr.Textbox(
                        label="📊 状态信息",
                        value=status_msg,
                    )

                    # FBX 下载区域
                    with gr.Row(visible=False) as self.fbx_download_row:
                        if getattr(self.runtime, "fbx_available", False):
                            self.fbx_files = gr.File(
                                label="📦 下载 FBX 文件",
                                file_count="multiple",
                                interactive=False,
                            )
                        else:
                            self.fbx_files = gr.State([])

                # 右侧显示区域
                with gr.Column(scale=3):
                    self.output_display = gr.HTML(
                        value=HTML_OUTPUT_PLACEHOLDER, show_label=False, elem_classes=["flask-display"]
                    )

            # 页脚
            gr.Markdown(FOOTER_MD, elem_classes=["footer"])

            self._bind_events()
            demo.load(fn=self._get_header_text, outputs=[self.header_md])
            return demo

    def _build_advanced_settings(self):
        # 仅在重写功能可用时显示重写选项
        if self.prompt_engineering_available:
            with gr.Group():
                gr.Markdown("### 🔄 文本重写选项")
                with gr.Row():
                    self.enable_rewrite = gr.Checkbox(
                        label="启用文本重写",
                        value=True,
                        info="自动优化文本提示词以获得更好的动作生成效果",
                    )

            with gr.Group():
                gr.Markdown("### ⏱️ 时长设置")
                self.enable_duration_est = gr.Checkbox(
                    label="启用时长估算",
                    value=True,
                    info="自动估算动作的时长",
                )
        else:
            # 创建带有默认值的隐藏占位符（禁用）
            self.enable_rewrite = gr.Checkbox(
                label="启用文本重写",
                value=False,
                visible=False,
            )
            self.enable_duration_est = gr.Checkbox(
                label="启用时长估算",
                value=False,
                visible=False,
            )
            with gr.Group():
                gr.Markdown("### ⚠️ 提示词工程不可用")
                gr.Markdown(
                    "文本重写和时长估算功能不可用。"
                    "将直接使用您输入的文本和时长。"
                )

        with gr.Group():
            gr.Markdown("### ⚙️ 生成参数")
            with gr.Row():
                with gr.Column(scale=3):
                    self.seed_input = gr.Textbox(
                        label="🎯 随机种子列表（逗号分隔）",
                        value="0,1,2,3",
                        placeholder="输入逗号分隔的种子列表（例如：0,1,2,3）",
                        info="随机种子控制生成动作的多样性",
                    )
                with gr.Column(scale=1, min_width=60, elem_classes=["dice-container"]):
                    self.dice_btn = gr.Button(
                        "🎲 幸运按钮",
                        variant="secondary",
                        size="sm",
                        elem_classes=["dice-button"],
                    )

            self.cfg_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5.0,
                step=0.1,
                label="⚙️ CFG 强度",
                info="文本保真度：越高越忠实于提示词",
            )

    def _bind_events(self):
        # 生成随机种子
        self.dice_btn.click(self._generate_random_seeds, outputs=[self.seed_input])

        # 绑定示例选择事件
        self.example_dropdown.change(
            fn=self._on_example_select,
            inputs=[self.example_dropdown],
            outputs=[self.text_input, self.seed_input, self.duration_slider],
        )

        # 重写文本逻辑（仅在重写功能可用时绑定）
        if self.prompt_engineering_available:
            self.rewrite_btn.click(fn=lambda: "正在重写文本，请稍候...", outputs=[self.status_output]).then(
                self._prompt_engineering,
                inputs=[
                    self.text_input,
                    self.duration_slider,
                    self.enable_rewrite,
                    self.enable_duration_est,
                ],
                outputs=[self.rewritten_text, self.generate_btn, self.duration_slider],
            ).then(
                fn=lambda rewritten: (
                    gr.update(visible=True),
                    f"✅ 文本重写完成！\n\n重写后的文本：\n{rewritten}\n\n您可以在上方进一步编辑，然后点击 [🚀 生成动作]",
                ),
                inputs=[self.rewritten_text],
                outputs=[self.rewritten_text, self.status_output],
            )

        # 生成动作逻辑
        self.generate_btn.click(
            fn=lambda: "正在生成动作，请稍候...（首次生成需要额外时间启动渲染器）",
            outputs=[self.status_output],
        ).then(
            self._generate_motion,
            inputs=[
                self.text_input,
                self.rewritten_text,
                self.seed_input,
                self.duration_slider,
                self.cfg_slider,
            ],
            outputs=[self.output_display, self.fbx_files],
            concurrency_limit=NUM_WORKERS,
        ).then(
            fn=lambda fbx_list: (
                (
                    "🎉 动作生成完成！您可以在右侧查看动作可视化结果。FBX 文件已准备好下载。"
                    if fbx_list
                    else "🎉 动作生成完成！您可以在右侧查看动作可视化结果"
                ),
                gr.update(visible=bool(fbx_list)),
            ),
            inputs=[self.fbx_files],
            outputs=[self.status_output, self.fbx_download_row],
        )

        # 重置逻辑 - 根据重写功能是否可用有不同行为
        if self.prompt_engineering_available:
            self.text_input.change(
                fn=lambda: (
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    "请先点击 [🔄 重写文本] 按钮重写文本",
                ),
                outputs=[self.rewritten_text, self.generate_btn, self.status_output],
            )
        else:
            # 当重写功能不可用时，输入文本后直接启用生成按钮
            self.text_input.change(
                fn=lambda text: (
                    gr.update(visible=False),
                    gr.update(interactive=bool(text.strip())),
                    (
                        "准备就绪！点击 [🚀 生成动作] 开始。"
                        if text.strip()
                        else "输入文本后直接点击 [🚀 生成动作]。"
                    ),
                ),
                inputs=[self.text_input],
                outputs=[self.rewritten_text, self.generate_btn, self.status_output],
            )
        # 仅在重写功能可用时绑定 rewritten_text 变更事件
        if self.prompt_engineering_available:
            self.rewritten_text.change(
                fn=lambda text: (
                    gr.update(interactive=bool(text.strip())),
                    (
                        "重写文本已修改，您可以点击 [🚀 生成动作]"
                        if text.strip()
                        else "重写文本不能为空，请输入有效文本"
                    ),
                ),
                inputs=[self.rewritten_text],
                outputs=[self.generate_btn, self.status_output],
            )


def create_demo(model_path: str):
    """创建 Gradio 演示，支持 Zero GPU。"""
    global _global_runtime, _global_args

    class Args:
        pass

    args = Args()
    args.model_path = model_path
    args.output_dir = "output/gradio"
    args.prompt_engineering_host = os.environ.get("PROMPT_HOST", None)
    args.prompt_engineering_model_path = os.environ.get("PROMPT_MODEL_PATH", LOCAL_PROMPTER_PATH)
    args.disable_prompt_engineering = os.environ.get("DISABLE_PROMPT_ENGINEERING", False)

    _global_args = args  # 设置全局 args 用于延迟加载

    # 检查必需文件:
    cfg = osp.join(args.model_path, "config.yml")
    ckpt = osp.join(args.model_path, "latest.ckpt")
    if not osp.exists(cfg):
        raise FileNotFoundError(f">>> 配置文件未找到: {cfg}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 对于 Zero GPU：启动时不加载模型，使用延迟加载
    # 为 UI 初始化创建一个最小运行时（不加载模型）
    if SPACES_AVAILABLE:
        print(">>> 检测到 Hugging Face Spaces。使用 Zero GPU 延迟加载。")
        print(">>> 模型将在首次 GPU 请求时加载。")

        # 为 UI 创建一个最小初始化的占位运行时
        class PlaceholderRuntime:
            def __init__(self):
                self.fbx_available = False
                self.prompt_engineering_host = args.prompt_engineering_host
                self.prompt_engineering_model_path = args.prompt_engineering_model_path

            def rewrite_text_and_infer_time(self, text: str):
                # 对于提示词重写，我们不需要 GPU
                from hymotion.prompt_engineering.prompt_rewrite import PromptRewriter

                rewriter = PromptRewriter(
                    host=self.prompt_engineering_host, model_path=self.prompt_engineering_model_path
                )
                return rewriter.rewrite_prompt_and_infer_time(text)

        runtime = PlaceholderRuntime()
    else:
        # 本地开发：立即加载模型
        print(">>> 检测到本地环境。在启动时加载模型。")
        skip_model_loading = False
        if not os.path.exists(ckpt):
            print(f">>> [警告] 检查点文件未找到: {ckpt}")
            print(f">>> [警告] 模型加载将被跳过。动作生成功能将不可用。")
            skip_model_loading = True

        print(">>> 正在初始化 T2MRuntime...")

        skip_text = False
        runtime = T2MRuntime(
            config_path=cfg,
            ckpt_name=ckpt,
            skip_text=skip_text,
            device_ids=None,
            skip_model_loading=skip_model_loading,
            disable_prompt_engineering=args.disable_prompt_engineering,
            prompt_engineering_host=args.prompt_engineering_host,
            prompt_engineering_model_path=args.prompt_engineering_model_path,
        )
        _global_runtime = runtime  # 为 GPU 函数设置全局运行时

    ui = T2MGradioUI(runtime=runtime, args=args)
    demo = ui.build_ui()
    return demo


if __name__ == "__main__":
    # 使用本地已下载的模型路径
    print(f">>> 使用本地模型路径: {LOCAL_MODEL_PATH}")
    demo = create_demo(LOCAL_MODEL_PATH)
    demo.launch(server_name="0.0.0.0")
