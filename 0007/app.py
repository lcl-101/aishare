# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

# ============================================================
# 修复 RMBG 模型与新版 transformers 的兼容性问题
# 必须在导入 actionmesh 之前执行
# ============================================================
from transformers import PreTrainedModel

# 保存原始的 __init__ 方法
_original_pretrained_init = PreTrainedModel.__init__

def _patched_pretrained_init(self, *args, **kwargs):
    """修补的初始化方法，确保 all_tied_weights_keys 被正确初始化为字典"""
    _original_pretrained_init(self, *args, **kwargs)
    # all_tied_weights_keys 必须是字典类型，支持 .keys(), .items(), .update() 等操作
    if not hasattr(self, 'all_tied_weights_keys') or self.all_tied_weights_keys is None:
        self.all_tied_weights_keys = {}
    # _tied_weights_keys 也应该是字典或 None
    if not hasattr(self, '_tied_weights_keys'):
        self._tied_weights_keys = None

# 只 patch 一次
if not getattr(PreTrainedModel, '_actionmesh_patched', False):
    PreTrainedModel.__init__ = _patched_pretrained_init
    PreTrainedModel._actionmesh_patched = True

import gradio as gr
import torch
from actionmesh.io.glb_export import create_animated_glb
from actionmesh.io.mesh_io import save_deformation, save_meshes
from actionmesh.io.video_input import load_frames
from actionmesh.pipeline import ActionMeshPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 创建符号链接，将 pretrained_weights 指向 checkpoints 目录
# 这样 pipeline.py 可以直接使用已下载的模型，无需重新下载
# ============================================================
def setup_model_symlinks():
    """创建模型目录的符号链接，避免重复下载"""
    base_dir = Path(__file__).parent
    checkpoints_dir = base_dir / "checkpoints"
    pretrained_dir = base_dir / "pretrained_weights"
    
    # 模型目录映射: pretrained_weights 中的名称 -> checkpoints 中的名称
    model_mappings = {
        "TripoSG": "TripoSG",
        "dinov2": "dinov2-large",
        "RMBG": "RMBG-1.4",
        "ActionMesh": "ActionMesh",
    }
    
    # 创建 pretrained_weights 目录（如果不存在）
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    for pretrained_name, checkpoint_name in model_mappings.items():
        source = checkpoints_dir / checkpoint_name
        target = pretrained_dir / pretrained_name
        
        if source.exists() and not target.exists():
            try:
                os.symlink(source.resolve(), target)
                logger.info(f"创建符号链接: {target} -> {source}")
            except OSError as e:
                logger.warning(f"无法创建符号链接 {target}: {e}")
        elif target.exists():
            logger.info(f"模型路径已存在: {target}")

# 启动时自动设置符号链接
setup_model_symlinks()

# 全局变量用于存储已加载的 pipeline
GLOBAL_PIPELINE = None

# ============================================================
# 自动下载并安装 Blender（用于导出动画 GLB）
# ============================================================
def setup_blender():
    """自动检测或下载 Blender 3.5.1"""
    import subprocess
    import tarfile
    
    base_dir = Path(__file__).parent
    blender_dir = base_dir / "blender-3.5.1-linux-x64"
    blender_exe = blender_dir / "blender"
    
    # 可能的 Blender 路径
    possible_paths = [
        blender_exe,
        Path("/usr/bin/blender"),
        Path("/opt/blender/blender"),
    ]
    
    # 检查是否已存在
    for bp in possible_paths:
        if bp.exists() and os.access(str(bp), os.X_OK):
            logger.info(f"检测到 Blender: {bp}")
            return str(bp)
    
    # 自动下载 Blender
    logger.info("未检测到 Blender，正在自动下载 Blender 3.5.1...")
    download_url = "https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz"
    tar_file = base_dir / "blender-3.5.1-linux-x64.tar.xz"
    
    try:
        # 下载
        subprocess.run(
            ["wget", "-q", "--show-progress", download_url, "-O", str(tar_file)],
            cwd=str(base_dir),
            check=True
        )
        logger.info("下载完成，正在解压...")
        
        # 解压
        with tarfile.open(tar_file, "r:xz") as tar:
            tar.extractall(path=base_dir)
        
        # 删除压缩包
        tar_file.unlink()
        
        if blender_exe.exists():
            logger.info(f"Blender 安装成功: {blender_exe}")
            return str(blender_exe)
        else:
            logger.warning("Blender 解压后未找到可执行文件")
            return None
            
    except Exception as e:
        logger.warning(f"Blender 自动下载失败: {e}")
        if tar_file.exists():
            tar_file.unlink()
        return None

# 启动时自动设置 Blender
DEFAULT_BLENDER_PATH = setup_blender()

def check_pytorch3d_installed() -> bool:
    """检查是否安装了 pytorch3d"""
    try:
        import pytorch3d
        return True
    except ImportError:
        logger.warning(
            "PyTorch3D 未安装。视频渲染将被跳过。"
        )
        return False


def check_blender_available(blender_path: str | None = None) -> bool:
    """检查 Blender 是否可用"""
    if blender_path is None:
        return False

    if os.path.isfile(blender_path) and os.access(blender_path, os.X_OK):
        return True
    else:
        return False


def init_pipeline(config_name: str = "actionmesh.yaml", dtype_str: str = "bfloat16", low_ram: bool = False):
    """初始化 ActionMesh pipeline"""
    global GLOBAL_PIPELINE
    
    if GLOBAL_PIPELINE is not None:
        logger.info("Pipeline 已经加载，跳过初始化")
        return GLOBAL_PIPELINE
    
    logger.info("正在初始化 ActionMesh Pipeline...")
    
    # 解析 dtype
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    
    # 初始化 pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dir = Path(__file__).parent / "actionmesh" / "configs"
    
    GLOBAL_PIPELINE = ActionMeshPipeline(
        config_name=config_name,
        config_dir=str(config_dir),
        dtype=dtype,
        lazy_loading=low_ram,
    )
    GLOBAL_PIPELINE.to(device)
    
    logger.info(f"Pipeline 初始化完成，使用设备: {device}")
    return GLOBAL_PIPELINE


@torch.no_grad()
def run_actionmesh(
    input_video,
    seed,
    blender_path,
    fast_mode,
    low_ram_mode,
    stage_0_steps,
    face_decimation,
    floaters_threshold,
    stage_1_steps,
    guidance_scales_str,
    anchor_idx,
    progress=gr.Progress()
):
    """运行 ActionMesh 推理"""
    try:
        progress(0, desc="准备中...")
        
        # 确定配置文件
        if fast_mode and low_ram_mode:
            config_name = "actionmesh_fast_lowram.yaml"
        elif fast_mode:
            config_name = "actionmesh_fast.yaml"
        elif low_ram_mode:
            config_name = "actionmesh_lowram.yaml"
        else:
            config_name = "actionmesh.yaml"
        
        # 初始化或获取 pipeline
        pipeline = init_pipeline(config_name=config_name, low_ram=low_ram_mode)
        
        # 创建临时输出目录
        output_dir = tempfile.mkdtemp(prefix="actionmesh_output_")
        
        # 处理输入视频路径
        if input_video is None:
            return None, None, None, "错误：请上传视频文件或图像文件夹"
        
        progress(0.1, desc="加载输入帧...")
        # 加载输入帧
        input_frames = load_frames(path=input_video, max_frames=31)
        
        progress(0.2, desc="运行推理（这可能需要几分钟）...")
        
        # 解析 guidance_scales
        guidance_scales = None
        if guidance_scales_str:
            try:
                guidance_scales = [float(x.strip()) for x in guidance_scales_str.split(",")]
            except:
                logger.warning(f"无法解析 guidance_scales: {guidance_scales_str}，使用默认值")
        
        # 运行推理
        meshes = pipeline(
            input=input_frames,
            seed=seed,
            stage_0_steps=stage_0_steps if stage_0_steps > 0 else None,
            face_decimation=face_decimation if face_decimation > 0 else None,
            floaters_threshold=floaters_threshold if floaters_threshold > 0 else None,
            stage_1_steps=stage_1_steps if stage_1_steps > 0 else None,
            guidance_scales=guidance_scales,
            anchor_idx=anchor_idx if anchor_idx >= 0 else None,
        )
        
        progress(0.7, desc="保存网格...")
        # 保存网格
        save_meshes(meshes, output_dir=output_dir)
        vertices_path, faces_path = save_deformation(
            meshes, path=f"{output_dir}/deformations"
        )
        
        # 结果文件路径
        mesh_output = None
        glb_output = None
        video_output = None
        first_mesh_preview = None
        
        # 查找生成的网格文件（优先 .glb，其次 .obj）
        glb_files = sorted(Path(output_dir).glob("mesh_*.glb"))
        obj_files = sorted(Path(output_dir).glob("*.obj"))
        
        if glb_files:
            # 使用第一个 mesh_00.glb 作为预览和下载
            mesh_output = str(glb_files[0])
            first_mesh_preview = str(glb_files[0])
        elif obj_files:
            mesh_output = str(obj_files[0])
            first_mesh_preview = str(obj_files[0])
        
        progress(0.8, desc="创建动画 GLB（如果提供了 Blender 路径）...")
        # 创建动画 GLB（如果有 Blender）
        if blender_path and check_blender_available(blender_path):
            animated_glb_path = f"{output_dir}/animated_mesh.glb"
            try:
                create_animated_glb(
                    blender_path=blender_path,
                    vertices_npy=vertices_path,
                    faces_npy=faces_path,
                    output_glb=animated_glb_path,
                    fps=8,
                )
                glb_output = animated_glb_path
            except Exception as e:
                logger.warning(f"创建动画 GLB 失败: {e}")
        
        progress(0.9, desc="渲染输出视频（如果安装了 PyTorch3D）...")
        # 渲染输出（如果有 pytorch3d）
        if check_pytorch3d_installed():
            try:
                from actionmesh.render.visualizer import ActionMeshVisualizer
                visualizer = ActionMeshVisualizer(image_size=256)
                visualizer.render(
                    meshes,
                    input_frames=input_frames.frames,
                    device=pipeline.device,
                    output_dir=output_dir,
                )
                # 查找渲染的视频
                video_files = list(Path(output_dir).glob("*.mp4"))
                if video_files:
                    video_output = str(video_files[0])
            except Exception as e:
                logger.warning(f"渲染视频失败: {e}")
        
        progress(1.0, desc="完成！")
        
        # 统计生成的网格数量
        all_mesh_files = sorted(Path(output_dir).glob("mesh_*.glb"))
        mesh_count = len(all_mesh_files)
        
        status_msg = f"处理完成！\n输出目录: {output_dir}\n生成网格数量: {mesh_count} 帧"
        if first_mesh_preview:
            status_msg += f"\n预览文件: {first_mesh_preview}"
        if glb_output:
            status_msg += f"\n动画 GLB: {glb_output}"
        if video_output:
            status_msg += f"\n渲染视频: {video_output}"
        
        # 返回值: mesh_output(下载), glb_output(下载), video_output, status_msg, model_preview, glb_preview
        return mesh_output, glb_output, video_output, status_msg, first_mesh_preview, glb_output
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return None, None, None, f"错误：{str(e)}", None, None


def create_ui():
    """创建 Gradio 界面"""
    
    # 定义示例数据
    EXAMPLES = {
        "davis_camel (骆驼)": "assets/examples/davis_camel",
        "davis_flamingo (火烈鸟)": "assets/examples/davis_flamingo",
        "kangaroo (袋鼠)": "assets/examples/kangaroo",
        "spring (弹簧)": "assets/examples/spring",
    }
    
    def get_example_images(example_name):
        """获取示例文件夹中的图片用于预览"""
        if not example_name or example_name not in EXAMPLES:
            return [], ""
        
        folder_path = EXAMPLES[example_name]
        base_dir = Path(__file__).parent
        full_path = base_dir / folder_path
        
        if not full_path.exists():
            return [], folder_path
        
        # 获取所有 PNG 图片并排序，显示全部
        images = sorted(full_path.glob("*.png"))
        images = [str(img) for img in images]
        
        return images, folder_path
    
    def on_example_change(example_name):
        """当示例选择变化时更新预览"""
        images, path = get_example_images(example_name)
        return images, path
    
    with gr.Blocks(title="ActionMesh - 视频转动画网格") as app:
        # YouTube 频道信息
        gr.Markdown(
            """
            # 🎬 ActionMesh - 视频转动画网格
            
            ### 📺 关注 [AI 技术分享频道](https://www.youtube.com/@rongyi-ai) 获取更多 AI 技术内容！
            
            ---
            
            将输入视频转换为动画 3D 网格。上传视频文件或包含 PNG 图像序列的文件夹。
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入设置")
                
                # 示例选择下拉框
                example_dropdown = gr.Dropdown(
                    choices=list(EXAMPLES.keys()),
                    label="选择示例",
                    info="选择一个内置示例进行测试",
                    value=None
                )
                
                # 图片预览画廊
                input_gallery = gr.Gallery(
                    label="输入图像预览",
                    columns=8,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )
                
                # 输入路径（可手动输入或由示例自动填充）
                input_video = gr.Textbox(
                    label="输入视频路径或图像文件夹路径",
                    placeholder="选择上方示例或手动输入路径",
                    info="输入视频文件路径（.mp4, .avi, .mov）或包含 PNG 图像序列的文件夹路径"
                )
                
                seed = gr.Number(
                    label="随机种子",
                    value=44,
                    precision=0
                )
                
                with gr.Accordion("⚙️ 高级设置", open=False):
                    fast_mode = gr.Checkbox(
                        label="快速模式",
                        value=False,
                        info="使用快速预设（stage_0_steps=50, stage_1_steps=15），质量可能略有降低"
                    )
                    
                    low_ram_mode = gr.Checkbox(
                        label="低内存模式",
                        value=False,
                        info="启用低内存优化，适合显存较小的 GPU"
                    )
                    
                    blender_path = gr.Textbox(
                        label="Blender 可执行文件路径",
                        value=DEFAULT_BLENDER_PATH or "",
                        placeholder="/usr/bin/blender",
                        info="已自动检测到 Blender" if DEFAULT_BLENDER_PATH else "提供 Blender 3.5.1 路径以导出动画 GLB 文件"
                    )
                    
                    stage_0_steps = gr.Number(
                        label="Stage 0 步数（图像转3D）",
                        value=0,
                        precision=0,
                        info="默认: 100，快速: 50，设为0使用默认值"
                    )
                    
                    face_decimation = gr.Number(
                        label="网格面数目标",
                        value=0,
                        precision=0,
                        info="网格简化的目标面数，默认: 40000，设为0使用默认值"
                    )
                    
                    floaters_threshold = gr.Number(
                        label="浮点清理阈值",
                        value=0,
                        info="移除浮点的阈值 (0.0-1.0)，默认: 0.02，设为0使用默认值"
                    )
                    
                    stage_1_steps = gr.Number(
                        label="Stage 1 步数（时序去噪）",
                        value=0,
                        precision=0,
                        info="默认: 30，快速: 15，设为0使用默认值"
                    )
                    
                    guidance_scales_str = gr.Textbox(
                        label="引导比例",
                        placeholder="7.5",
                        info="无分类器引导比例，多个值用逗号分隔，留空使用默认值 [7.5]"
                    )
                    
                    anchor_idx = gr.Number(
                        label="锚点帧索引",
                        value=-1,
                        precision=0,
                        info="固定拓扑的锚点帧索引，默认: 0，设为-1使用默认值"
                    )
                
                run_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输出结果")
                
                status_output = gr.Textbox(
                    label="状态信息",
                    lines=4,
                    interactive=False
                )
                
                # 3D 模型预览
                with gr.Tabs():
                    with gr.TabItem("🎮 3D 模型预览"):
                        model_preview = gr.Model3D(
                            label="3D 网格预览",
                            height=400,
                            interactive=False
                        )
                    
                    with gr.TabItem("📦 GLB 动画"):
                        glb_preview = gr.Model3D(
                            label="动画 GLB 预览（需要 Blender 导出）",
                            height=400,
                            interactive=False
                        )
                    
                    with gr.TabItem("🎬 渲染视频"):
                        video_output = gr.Video(
                            label="渲染视频（需要 PyTorch3D）",
                            interactive=False,
                            height=400
                        )
                
                gr.Markdown("### 📁 下载文件")
                with gr.Row():
                    mesh_output = gr.File(
                        label="网格文件 (.obj)",
                        interactive=False
                    )
                    
                    glb_output = gr.File(
                        label="动画 GLB 文件",
                        interactive=False
                    )
        
        # 示例选择事件
        example_dropdown.change(
            fn=on_example_change,
            inputs=[example_dropdown],
            outputs=[input_gallery, input_video]
        )
        
        # 绑定按钮事件
        run_btn.click(
            fn=run_actionmesh,
            inputs=[
                input_video,
                seed,
                blender_path,
                fast_mode,
                low_ram_mode,
                stage_0_steps,
                face_decimation,
                floaters_threshold,
                stage_1_steps,
                guidance_scales_str,
                anchor_idx
            ],
            outputs=[mesh_output, glb_output, video_output, status_output, model_preview, glb_preview]
        )
        
        gr.Markdown(
            """
            ---
            
            ### 📝 使用说明
            
            1. **选择示例**: 从下拉框选择一个内置示例，可以预览输入图像
            2. **或手动输入**: 在路径输入框中输入视频文件或图像文件夹路径
            3. **配置参数**: 根据需要调整参数，或使用默认设置
            4. **开始生成**: 点击"开始生成"按钮开始处理
            5. **预览结果**: 在右侧查看 3D 模型预览、GLB 动画或渲染视频
            6. **下载文件**: 下载生成的网格文件和 GLB 文件
            
            **注意**: 
            - 处理可能需要几分钟，取决于视频长度和硬件配置
            - 提供 Blender 路径可以导出可在 Blender 中导入的动画网格文件
            - 如果安装了 PyTorch3D，会自动生成渲染视频
            
            ### 🔗 相关链接
            
            - [YouTube 频道: AI 技术分享频道](https://www.youtube.com/@rongyi-ai)
            - [ActionMesh 项目](https://github.com/facebookresearch/ActionMesh)
            """
        )
    
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ActionMesh Gradio Web App")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--server_port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共分享链接")
    parser.add_argument("--preload", action="store_true", help="启动时预加载模型")
    args = parser.parse_args()
    
    # 预加载模型（如果指定）
    if args.preload:
        logger.info("预加载模型中...")
        init_pipeline()
        logger.info("模型预加载完成")
    
    # 创建并启动应用
    app = create_ui()
    app.queue()
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        theme=gr.themes.Soft()
    )
