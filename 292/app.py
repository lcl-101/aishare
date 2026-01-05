"""
UltraShape Gradio Web Application
基于 Gradio 的 UltraShape 3D 网格精炼 Web 界面
"""

import os
import sys
import tempfile
import torch
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ultrashape.rembg import BackgroundRemover
from ultrashape.utils.misc import instantiate_from_config
from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
from ultrashape.utils import voxelize_from_point
from ultrashape.pipelines import UltraShapePipeline

# 全局变量存储模型
pipeline = None
loader = None
rembg = None
token_num = None
voxel_res = None
device = None


def load_models():
    """启动时加载模型"""
    global pipeline, loader, rembg, token_num, voxel_res, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(project_root, "configs/infer_dit_refine.yaml")
    ckpt_path = os.path.join(project_root, "checkpoints/UltraShape/ultrashape_v1.pt")
    
    print("=" * 60)
    print("正在加载 UltraShape 模型...")
    print("=" * 60)
    
    print(f"正在从 {config_path} 加载配置...")
    config = OmegaConf.load(config_path)
    
    # 修改 dinov2 模型路径为本地路径
    dinov2_local_path = os.path.join(project_root, "checkpoints/dinov2-large")
    if os.path.exists(dinov2_local_path):
        config.model.params.conditioner_config.params.main_image_encoder.kwargs.version = dinov2_local_path
        print(f"使用本地 DINOv2 模型: {dinov2_local_path}")
    
    print("正在初始化 VAE...")
    vae = instantiate_from_config(config.model.params.vae_config)
    
    print("正在初始化 DiT...")
    dit = instantiate_from_config(config.model.params.dit_cfg)
    
    print("正在初始化图像编码器...")
    conditioner = instantiate_from_config(config.model.params.conditioner_config)
    
    print("正在初始化调度器和处理器...")
    scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
    image_processor = instantiate_from_config(config.model.params.image_processor_cfg)
    
    print(f"正在从 {ckpt_path} 加载权重...")
    weights = torch.load(ckpt_path, map_location='cpu')
    
    vae.load_state_dict(weights['vae'], strict=True)
    dit.load_state_dict(weights['dit'], strict=True)
    conditioner.load_state_dict(weights['conditioner'], strict=True)
    
    vae.eval().to(device)
    dit.eval().to(device)
    conditioner.eval().to(device)
    
    if hasattr(vae, 'enable_flashvdm_decoder'):
        vae.enable_flashvdm_decoder()
    
    pipeline = UltraShapePipeline(
        vae=vae,
        model=dit,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor
    )
    
    token_num = config.model.params.vae_config.params.num_latents
    voxel_res = config.model.params.vae_config.params.voxel_query_res
    
    print(f"正在初始化表面加载器 (Token 数量: {token_num})...")
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )
    
    print("正在初始化背景移除器...")
    rembg = BackgroundRemover()
    
    print("=" * 60)
    print("✅ 模型加载完成！")
    print("=" * 60)


def refine_mesh(
    image_input,
    mesh_input,
    steps: int = 50,
    scale: float = 0.99,
    octree_res: int = 1024,
    seed: int = 42,
    remove_bg: bool = True,
    progress=gr.Progress()
):
    """
    执行网格精炼推理
    """
    global pipeline, loader, rembg, token_num, voxel_res, device
    
    if pipeline is None:
        return None, "❌ 错误：模型尚未加载，请稍后重试"
    
    if image_input is None:
        return None, "❌ 错误：请上传参考图像"
    
    if mesh_input is None:
        return None, "❌ 错误：请上传粗糙网格文件"
    
    try:
        progress(0.1, desc="正在处理输入图像...")
        
        # 处理图像
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = Image.fromarray(image_input)
        
        if remove_bg or image.mode != 'RGBA':
            progress(0.2, desc="正在移除背景...")
            image = rembg(image)
        
        progress(0.3, desc="正在加载网格...")
        
        # 处理网格文件路径
        mesh_path = mesh_input
        if hasattr(mesh_input, 'name'):
            mesh_path = mesh_input.name
        
        surface = loader(mesh_path, normalize_scale=scale).to(device, dtype=torch.float16)
        pc = surface[:, :, :3]  # [B, N, 3]
        
        progress(0.4, desc="正在体素化...")
        _, voxel_idx = voxelize_from_point(pc, token_num, resolution=voxel_res)
        
        # 如果体素数量少于目标数量，通过重复填充到目标大小
        if voxel_idx.shape[1] < token_num:
            B, actual_k, _ = voxel_idx.shape
            repeat_times = (token_num + actual_k - 1) // actual_k
            voxel_idx = voxel_idx.repeat(1, repeat_times, 1)[:, :token_num, :]
        
        progress(0.5, desc="正在运行扩散过程...")
        generator = torch.Generator(device).manual_seed(seed)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh, _ = pipeline(
                image=image,
                voxel_cond=voxel_idx,
                generator=generator,
                box_v=1.0,
                mc_level=0.0,
                octree_resolution=octree_res,
                num_inference_steps=steps,
            )
        
        progress(0.9, desc="正在保存结果...")
        
        # 保存结果
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, "refined_mesh.glb")
        
        mesh = mesh[0]
        mesh.export(output_path)
        
        progress(1.0, desc="完成！")
        
        return output_path, f"✅ 精炼完成！推理步数: {steps}, 随机种子: {seed}"
    
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理过程中发生错误：{str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def create_ui():
    """创建 Gradio 界面"""
    
    # 自定义 CSS
    custom_css = """
    .youtube-banner {
        background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%);
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .youtube-banner a {
        color: white !important;
        text-decoration: none;
        font-size: 18px;
        font-weight: bold;
    }
    .youtube-banner a:hover {
        text-decoration: underline;
    }
    .youtube-icon {
        margin-right: 8px;
    }
    """
    
    with gr.Blocks(
        title="UltraShape - 3D 网格精炼",
        css=custom_css,
        theme=gr.themes.Soft()
    ) as demo:
        
        # YouTube 频道横幅
        gr.HTML("""
        <div class="youtube-banner">
            <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">
                <span class="youtube-icon">▶️</span>
                AI 技术分享频道 - 欢迎订阅我的 YouTube 频道！
            </a>
        </div>
        """)
        
        gr.Markdown("""
        # 🎨 UltraShape - 3D 网格精炼工具
        
        上传参考图像和粗糙网格，使用 UltraShape 模型生成精细的 3D 网格。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输入")
                
                image_input = gr.Image(
                    label="参考图像",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=300
                )
                
                mesh_input = gr.Model3D(
                    label="粗糙网格文件 (.glb / .obj)",
                    height=300,
                    clear_color=[0.8, 0.8, 0.8, 1.0]
                )
                
                gr.Markdown("### ⚙️ 参数设置")
                
                with gr.Row():
                    steps = gr.Slider(
                        label="推理步数",
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=1,
                        info="更多步数可能产生更好的结果，但需要更长时间"
                    )
                    
                    seed = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0,
                        info="设置随机种子以获得可复现的结果"
                    )
                
                with gr.Row():
                    scale = gr.Slider(
                        label="网格归一化比例",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.99,
                        step=0.01,
                        info="网格归一化的缩放因子"
                    )
                    
                    octree_res = gr.Slider(
                        label="八叉树分辨率",
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=64,
                        info="Marching Cubes 分辨率"
                    )
                
                remove_bg = gr.Checkbox(
                    label="移除背景",
                    value=True,
                    info="自动移除图像背景"
                )
                
                submit_btn = gr.Button(
                    "🚀 开始精炼",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输出")
                
                output_model = gr.Model3D(
                    label="精炼后的 3D 模型",
                    height=400,
                    clear_color=[0.8, 0.8, 0.8, 1.0]
                )
                
                output_file = gr.File(
                    label="下载精炼后的网格文件"
                )
                
                status_text = gr.Textbox(
                    label="状态",
                    interactive=False,
                    lines=2
                )
        
        # 使用说明
        gr.Markdown("""
        ---
        ### 📖 使用说明
        
        1. **上传参考图像**：上传您想要作为参考的图像（支持 PNG、JPG 等格式）
        2. **上传粗糙网格**：上传需要精炼的粗糙 3D 网格文件（支持 .glb、.obj、.ply 格式）
        3. **调整参数**（可选）：
           - **推理步数**：更多步数通常产生更好的结果
           - **随机种子**：用于可复现的结果
           - **网格归一化比例**：调整网格的缩放
           - **八叉树分辨率**：控制输出网格的精度
        4. **点击"开始精炼"**：等待处理完成
        5. **查看和下载结果**：在右侧预览 3D 模型并下载
        
        ---
        ### ⚠️ 注意事项
        
        - 推理过程需要 GPU，请确保有足够的显存
        - 处理时间取决于推理步数和八叉树分辨率
        - 建议使用具有清晰背景的参考图像以获得最佳效果
        """)
        
        # 绑定事件
        def process_and_return(image, mesh, steps, scale, octree_res, seed, remove_bg, progress=gr.Progress()):
            output_path, status = refine_mesh(
                image, mesh, int(steps), scale, int(octree_res), int(seed), remove_bg, progress
            )
            return output_path, output_path, status
        
        submit_btn.click(
            fn=process_and_return,
            inputs=[image_input, mesh_input, steps, scale, octree_res, seed, remove_bg],
            outputs=[output_model, output_file, status_text]
        )
    
    return demo


if __name__ == "__main__":
    # 启动时加载模型
    load_models()
    
    # 创建并启动界面
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
