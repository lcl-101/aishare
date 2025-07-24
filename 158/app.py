import gradio as gr
import os
import sys
import time
import tempfile
from typing import Any, Union
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from accelerate.utils import set_seed

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, make_grid_for_images_or_videos, export_renderings
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

# Global variables to store loaded models
pipe = None
rmbg_net = None
models_loaded = False

def load_models():
    """加载模型（如果尚未加载）"""
    global pipe, rmbg_net, models_loaded
    
    if models_loaded:
        # 如果模型已加载，也显示当前VRAM使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            return f"✅ 模型已加载！设备：cuda\n💾 VRAM: {memory_allocated:.1f}GB / {gpu_memory:.1f}GB (已分配)\n📊 缓存: {memory_reserved:.1f}GB (已预留)"
        else:
            return "✅ 模型已加载！设备：cpu"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        # 显示加载前的VRAM状态
        if device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            initial_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
            
        # 下载预训练权重
        partcrafter_weights_dir = "pretrained_weights/PartCrafter"
        rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
        
        snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
        
        # 初始化模型
        rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        rmbg_net.eval()
        
        pipe = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)
        
        models_loaded = True
        
        # 显示加载后的VRAM状态
        if device == "cuda":
            final_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            model_memory = final_memory - initial_memory
            
            return f"✅ 模型加载成功！设备：{device}\n💾 VRAM: {final_memory:.1f}GB / {gpu_memory:.1f}GB (已分配)\n📊 缓存: {memory_reserved:.1f}GB (已预留)\n🚀 模型占用: {model_memory:.1f}GB"
        else:
            return f"✅ 模型加载成功！设备：{device}"
        
    except Exception as e:
        return f"❌ 模型加载失败：{str(e)}"

@torch.no_grad()
def run_partcrafter(
    image_input: Image.Image,
    num_parts: int,
    seed: int,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    use_rmbg: bool = False,
    progress=gr.Progress()
):
    """运行PartCrafter推理"""
    global pipe, rmbg_net, models_loaded
    
    if not models_loaded:
        return None, None, None, "❌ 请先加载模型！"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        progress(0.1, desc="正在预处理图像...")
        
        # 预处理图像
        if use_rmbg:
            # 为RMBG创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image_input.save(tmp_file.name)
                img_pil = prepare_image(tmp_file.name, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
                # 清理临时文件
                os.unlink(tmp_file.name)
        else:
            img_pil = image_input
        
        progress(0.2, desc="正在准备生成...")
        set_seed(seed)
        
        progress(0.3, desc="正在生成3D部件...")
        start_time = time.time()
        
        outputs = pipe(
            image=[img_pil] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=num_tokens,
            generator=torch.Generator(device=pipe.device).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_num_expanded_coords=int(1e9),
            use_flash_decoder=False,
        ).meshes
        
        end_time = time.time()
        generation_time = f"⏱️ 生成耗时：{end_time - start_time:.2f} 秒"
        
        progress(0.8, desc="正在处理网格...")
        
        # 处理空网格
        for i in range(len(outputs)):
            if outputs[i] is None:
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        
        # 创建合并网格
        merged_mesh = get_colored_mesh_composition(outputs)
        
        progress(0.9, desc="正在保存文件...")
        
        # 创建临时输出目录
        output_dir = tempfile.mkdtemp(prefix="partcrafter_")
        
        # 保存各个部件
        part_files = []
        for i, mesh in enumerate(outputs):
            part_file = os.path.join(output_dir, f"part_{i:02d}.glb")
            mesh.export(part_file)
            part_files.append(part_file)
        
        # 保存合并对象
        object_file = os.path.join(output_dir, "object.glb")
        merged_mesh.export(object_file)
        
        # 尝试渲染
        rendered_gif = None
        try:
            progress(0.95, desc="正在渲染预览...")
            
            num_views = 36
            radius = 4
            fps = 18
            
            rendered_images = render_views_around_mesh(
                merged_mesh,
                num_views=num_views,
                radius=radius,
            )
            
            # 保存渲染为GIF
            gif_file = os.path.join(output_dir, "rendering.gif")
            export_renderings(rendered_images, gif_file, fps=fps)
            rendered_gif = gif_file
            
        except Exception as render_e:
            print(f"渲染失败：{render_e}")
            rendered_gif = None
        
        progress(1.0, desc="完成！")
        
        success_message = f"✅ 成功生成 {num_parts} 个部件！\n{generation_time}"
        
        return object_file, part_files, rendered_gif, success_message
        
    except Exception as e:
        return None, None, None, f"❌ 生成过程中出错：{str(e)}"

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(
        title="PartCrafter 中文界面", 
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 2rem; }
        .model-status { padding: 1rem; border-radius: 8px; margin: 1rem 0; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🛠️ PartCrafter 中文界面
        
        **从单张图片生成可分离部件的3D对象！**
        
        上传一张图片，PartCrafter将把它分解成多个可以组装在一起的3D部件。
        """, elem_classes=["header"])
        
        # 模型加载部分
        with gr.Row():
            with gr.Column(scale=1):
                load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
                model_status = gr.Textbox(
                    label="模型状态", 
                    value="🟡 模型未加载。点击'加载模型'开始加载。",
                    interactive=False,
                    elem_classes=["model-status"]
                )
        
        gr.Markdown("---")
        
        # 主界面
        with gr.Row():
            # 输入列
            with gr.Column(scale=1):
                gr.Markdown("## 📤 输入")
                
                input_image = gr.Image(
                    label="上传图片",
                    type="pil",
                    height=300
                )
                
                with gr.Group():
                    gr.Markdown("### ⚙️ 生成设置")
                    
                    num_parts = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=3,
                        step=1,
                        label="部件数量",
                        info="将对象分解成多少个部件"
                    )
                    
                    seed = gr.Number(
                        value=42,
                        label="随机种子",
                        info="用于生成可重现结果的随机种子"
                    )
                    
                    with gr.Accordion("🔧 高级设置", open=False):
                        num_tokens = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=256,
                            label="令牌数量",
                            info="数值越高 = 细节越多（速度越慢）"
                        )
                        
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=10,
                            label="推理步数",
                            info="步数越多 = 质量越好（速度越慢）"
                        )
                        
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            value=7.0,
                            step=0.5,
                            label="引导强度",
                            info="多大程度上遵循输入图像"
                        )
                        
                        use_rmbg = gr.Checkbox(
                            label="显式去背景处理",
                            value=False,
                            info="对复杂背景图像推荐开启，简单背景可关闭"
                        )
                
                generate_btn = gr.Button("🚀 生成3D部件", variant="primary", size="lg")
        
            # 输出列
            with gr.Column(scale=1):
                gr.Markdown("## 📥 输出")
                
                status_output = gr.Textbox(
                    label="状态",
                    value="准备生成！",
                    interactive=False
                )
                
                with gr.Tab("🎬 预览"):
                    rendered_output = gr.Image(
                        label="360° 旋转预览",
                        type="filepath",
                        height=300
                    )
                
                with gr.Tab("📁 下载"):
                    object_output = gr.File(
                        label="完整对象 (object.glb)",
                        file_types=[".glb"]
                    )
                    
                    parts_output = gr.File(
                        label="各个部件",
                        file_count="multiple",
                        file_types=[".glb"]
                    )
        
        # 示例图片
        gr.Markdown("## 🎯 试试这些示例")
        
        example_images = [
            ["assets/images/np3_2f6ab901c5a84ed6bbdf85a67b22a2ee.png", 3, 42],
            ["assets/images/np4_2444ea17f3a448b1bb7e2a74b276f015.png", 4, 123],
            ["assets/images/np5_23ae06bb5cf84e13ae973721fa5f5625.png", 5, 456],
        ]
        
        # 检查示例图片是否存在
        available_examples = []
        for img_path, parts, seed_val in example_images:
            if os.path.exists(img_path):
                available_examples.append([img_path, parts, seed_val])
        
        if available_examples:
            gr.Examples(
                examples=available_examples,
                inputs=[input_image, num_parts, seed],
                label="点击示例加载"
            )
        
        # 事件处理
        load_btn.click(
            fn=load_models,
            outputs=[model_status]
        )
        
        generate_btn.click(
            fn=run_partcrafter,
            inputs=[
                input_image, num_parts, seed, num_tokens, 
                num_inference_steps, guidance_scale, use_rmbg
            ],
            outputs=[object_output, parts_output, rendered_output, status_output]
        )
        
        # 页脚
        gr.Markdown("""
        ---
        
        💡 **使用技巧：**
        - 建议从3-5个部件开始，效果最佳
        - 更高的质量设置需要更长时间但产生更好的结果  
        - 对于复杂背景的图像使用背景移除功能
        - 生成的.glb文件可以在Blender、Unity或任何3D查看器中打开
        
        🎥 **专为YouTube创作者优化：** 这个工具非常适合展示AI驱动的3D生成技术！
        """)
    
    return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_interface()
    
    # 启动，设置share=True可获得公开URL用于YouTube演示
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=False,  # 设为True可获得公开URL
        debug=True
    )
