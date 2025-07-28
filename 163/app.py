#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gradio as gr
import shutil
import tempfile
from pathlib import Path
import time
import gc
import torch
import psutil
import numpy as np

# Import the demo classes
from demo_panogen import Text2PanoramaDemo, Image2PanoramaDemo
from demo_scenegen import HYworldDemo

# Global variables to store loaded models
text2pano_demo = None
image2pano_demo = None
scenegen_demo = None

def load_model_on_demand(model_type):
    """按需加载模型以节省内存"""
    global text2pano_demo, image2pano_demo, scenegen_demo
    
    try:
        # 加载前检查内存
        check_and_clean_memory(threshold_percent=80)
        
        if model_type == "text2pano" and text2pano_demo is None:
            print("📝 加载文本到全景图模型...")
            print(f"📊 加载前内存状态: {get_memory_info()}")
            text2pano_demo = Text2PanoramaDemo()
            print(f"📊 加载后内存状态: {get_memory_info()}")
            
        elif model_type == "image2pano" and image2pano_demo is None:
            print("🖼️ 加载图像到全景图模型...")
            print(f"📊 加载前内存状态: {get_memory_info()}")
            image2pano_demo = Image2PanoramaDemo()
            print(f"📊 加载后内存状态: {get_memory_info()}")
            
        elif model_type == "scenegen" and scenegen_demo is None:
            print("🌍 加载场景生成模型...")
            print(f"📊 加载前内存状态: {get_memory_info()}")
            scenegen_demo = HYworldDemo()
            print(f"📊 加载后内存状态: {get_memory_info()}")
            
        return True
    except Exception as e:
        print(f"❌ {model_type} 模型加载失败: {str(e)}")
        return False

def clear_unused_models():
    """清理未使用的模型以释放内存"""
    global text2pano_demo, image2pano_demo, scenegen_demo
    
    try:
        print("🧹 开始清理模型和内存...")
        
        # 清理全景图模型
        if text2pano_demo is not None:
            if hasattr(text2pano_demo, 'pipe'):
                del text2pano_demo.pipe
            del text2pano_demo
            text2pano_demo = None
            print("📝 文本到全景图模型已清理")
            
        if image2pano_demo is not None:
            if hasattr(image2pano_demo, 'pipe'):
                del image2pano_demo.pipe
            del image2pano_demo
            image2pano_demo = None
            print("🖼️ 图像到全景图模型已清理")
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🚮 CUDA缓存已清理")
        
        print(f"✅ 模型清理完成: {get_memory_info()}")
        return True
    except Exception as e:
        print(f"⚠️ 清理模型时出现警告: {str(e)}")
        return False

def get_memory_info():
    """获取系统内存信息"""
    try:
        memory = psutil.virtual_memory()
        gpu_info = ""
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_used = torch.cuda.memory_allocated(0) / 1024**3
            gpu_info = f"GPU: {gpu_used:.1f}GB / {gpu_memory:.1f}GB"
        
        return f"内存: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%) | {gpu_info}"
    except:
        return "内存信息获取失败"

def check_and_clean_memory(threshold_percent=85):
    """检查内存使用率，超过阈值时自动清理"""
    try:
        memory = psutil.virtual_memory()
        if memory.percent > threshold_percent:
            print(f"⚠️ 内存使用率过高 ({memory.percent:.1f}%)，开始自动清理...")
            clear_unused_models()
            return True
        return False
    except:
        return False

def load_models():
    """预加载基础模型（可选）"""
    print("🚀 HunyuanWorld 1.0 Web Demo 已启动")
    print("💡 模型将在首次使用时自动加载以节省内存")
    print(f"📊 {get_memory_info()}")
    return True

def get_example_cases():
    """获取所有示例案例的信息"""
    examples_dir = Path("examples")
    cases = {}
    
    for case_dir in sorted(examples_dir.glob("case*")):
        case_name = case_dir.name
        case_info = {
            "name": case_name,
            "has_image": (case_dir / "input.png").exists(),
            "has_prompt": (case_dir / "prompt.txt").exists(),
            "classes": "",
            "labels_fg1": "",
            "labels_fg2": "",
            "prompt": ""
        }
        
        # 读取配置文件
        if (case_dir / "classes.txt").exists():
            case_info["classes"] = (case_dir / "classes.txt").read_text().strip()
        
        if (case_dir / "labels_fg1.txt").exists():
            case_info["labels_fg1"] = (case_dir / "labels_fg1.txt").read_text().strip()
            
        if (case_dir / "labels_fg2.txt").exists():
            case_info["labels_fg2"] = (case_dir / "labels_fg2.txt").read_text().strip()
            
        if (case_dir / "prompt.txt").exists():
            case_info["prompt"] = (case_dir / "prompt.txt").read_text().strip()
            
        cases[case_name] = case_info
    
    return cases

def load_example_case(case_name):
    """加载选定的示例案例"""
    if not case_name:
        return None, "", "", "", "", ""
    
    cases = get_example_cases()
    if case_name not in cases:
        return None, "", "", "", "", ""
    
    case_info = cases[case_name]
    
    # 返回图像路径（如果存在）
    image_path = f"examples/{case_name}/input.png" if case_info["has_image"] else None
    
    return (
        image_path,
        case_info["prompt"],
        case_info["classes"],
        case_info["labels_fg1"],
        case_info["labels_fg2"],
        "Image to World" if case_info["has_image"] else "Text to World"
    )

def create_3d_viewer(ply_files, output_dir):
    """创建3D模型查看器HTML"""
    import base64
    import os
    
    # 读取PLY文件并转换为base64（用于嵌入到HTML中）
    ply_data_list = []
    for ply_file in ply_files:
        layer_name = os.path.basename(ply_file).replace('.ply', '')
        with open(ply_file, 'rb') as f:
            ply_data = base64.b64encode(f.read()).decode('utf-8')
            ply_data_list.append({
                'name': layer_name,
                'data': ply_data
            })
    
    # 生成图层按钮
    layer_buttons = []
    for ply in ply_data_list:
        layer_buttons.append(f'<button class="control-btn" onclick="toggleLayer(\'{ply["name"]}\')">{ply["name"]}</button>')
    layer_buttons_html = ' '.join(layer_buttons)
    
    # 生成HTML查看器
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>3D World Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>
    <style>
        body {{ 
            margin: 0; 
            font-family: Arial, sans-serif;
            background: #222;
            color: white;
        }}
        #container {{
            width: 100%;
            height: 500px;
            position: relative;
        }}
        #controls {{
            padding: 10px;
            background: #333;
            border-bottom: 1px solid #444;
        }}
        .control-btn {{
            padding: 8px 12px;
            margin-right: 5px;
            border: none;
            border-radius: 4px;
            background: #555;
            color: white;
            cursor: pointer;
        }}
        .control-btn:hover {{
            background: #666;
        }}
        .control-btn.active {{
            background: #4CAF50;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="controls">
        <button class="control-btn" onclick="resetCamera()">重置视角</button>
        <button class="control-btn" onclick="toggleRotation()">自动旋转</button>
        <button class="control-btn" onclick="toggleWireframe()">线框模式</button>
        {layer_buttons_html}
    </div>
    <div id="container"></div>
    <div id="info">
        <div>使用鼠标拖拽旋转视角</div>
        <div>鼠标滚轮缩放</div>
        <div>WASD键移动</div>
    </div>

    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 500, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, 500);
        renderer.setClearColor(0x222222);
        document.getElementById('container').appendChild(renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // Controls
        let isRotating = true;
        let meshes = {{}};
        
        // Load PLY data
        const plyData = {ply_data_list};
        const loader = new THREE.PLYLoader();
        
        plyData.forEach(plyInfo => {{
            const binaryString = atob(plyInfo.data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {{
                bytes[i] = binaryString.charCodeAt(i);
            }}
            
            const geometry = loader.parse(bytes.buffer);
            const material = new THREE.MeshLambertMaterial({{ 
                color: Math.random() * 0xffffff,
                side: THREE.DoubleSide
            }});
            const mesh = new THREE.Mesh(geometry, material);
            
            meshes[plyInfo.name] = mesh;
            scene.add(mesh);
        }});

        // Center and scale the scene
        const box = new THREE.Box3().setFromObject(scene);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        scene.position.sub(center);
        camera.position.set(0, 0, Math.max(size.x, size.y, size.z) * 2);

        // Mouse controls
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        
        renderer.domElement.addEventListener('mousedown', (event) => {{
            mouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        }});
        
        renderer.domElement.addEventListener('mouseup', () => {{
            mouseDown = false;
        }});
        
        renderer.domElement.addEventListener('mousemove', (event) => {{
            if (!mouseDown) return;
            
            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;
            
            scene.rotation.y += deltaX * 0.01;
            scene.rotation.x += deltaY * 0.01;
            
            mouseX = event.clientX;
            mouseY = event.clientY;
        }});
        
        // Zoom
        renderer.domElement.addEventListener('wheel', (event) => {{
            camera.position.z += event.deltaY * 0.01;
            camera.position.z = Math.max(0.1, camera.position.z);
        }});

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            
            if (isRotating) {{
                scene.rotation.y += 0.005;
            }}
            
            renderer.render(scene, camera);
        }}
        animate();

        // Control functions
        function resetCamera() {{
            camera.position.set(0, 0, Math.max(size.x, size.y, size.z) * 2);
            scene.rotation.set(0, 0, 0);
        }}
        
        function toggleRotation() {{
            isRotating = !isRotating;
        }}
        
        function toggleWireframe() {{
            Object.values(meshes).forEach(mesh => {{
                mesh.material.wireframe = !mesh.material.wireframe;
            }});
        }}
        
        function toggleLayer(layerName) {{
            if (meshes[layerName]) {{
                meshes[layerName].visible = !meshes[layerName].visible;
            }}
        }}
        
        // Keyboard controls
        const keys = {{ w: false, a: false, s: false, d: false }};
        
        document.addEventListener('keydown', (event) => {{
            switch (event.key.toLowerCase()) {{
                case 'w': keys.w = true; break;
                case 'a': keys.a = true; break;
                case 's': keys.s = false; break;
                case 'd': keys.d = true; break;
            }}
        }});
        
        document.addEventListener('keyup', (event) => {{
            switch (event.key.toLowerCase()) {{
                case 'w': keys.w = false; break;
                case 'a': keys.a = false; break;
                case 's': keys.s = false; break;
                case 'd': keys.d = false; break;
            }}
        }});
        
        setInterval(() => {{
            if (keys.w) camera.position.z -= 0.1;
            if (keys.s) camera.position.z += 0.1;
            if (keys.a) camera.position.x -= 0.1;
            if (keys.d) camera.position.x += 0.1;
            camera.position.z = Math.max(0.1, camera.position.z);
        }}, 16);
    </script>
</body>
</html>'''
    
    # 保存HTML文件
    viewer_path = os.path.join(output_dir, "3d_viewer.html")
    with open(viewer_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return viewer_path

def create_3d_viewer_content(ply_files):
    """创建用于Gradio HTML组件的3D查看器内容"""
    layer_info = []
    for ply_file in ply_files:
        layer_name = os.path.basename(ply_file).replace('.ply', '')
        layer_info.append(f"• {layer_name}")
    
    layer_list = "<br>".join(layer_info)
    
    html_content = f'''
    <div style="text-align: center; padding: 20px; background: #f0f0f0; border-radius: 8px;">
        <h3 style="color: #333; margin-bottom: 15px;">🌍 3D世界生成成功！</h3>
        <div style="margin-bottom: 15px; color: #666;">
            <strong>生成的图层：</strong><br>
            {layer_list}
        </div>
        <div style="margin-bottom: 15px; padding: 10px; background: #e8f5e8; border-radius: 4px; color: #2e7d32;">
            <strong>💡 如何查看3D模型：</strong><br>
            1. 下载上方的PLY文件<br>
            2. 打开项目根目录的 <code>modelviewer.html</code><br>
            3. 上传PLY文件即可交互式预览
        </div>
        <div style="font-size: 12px; color: #999;">
            支持鼠标旋转、缩放和键盘控制（WASD移动）
        </div>
    </div>
    '''
    
    return html_content

def generate_panorama(mode, prompt, negative_prompt, input_image, seed, case_name):
    """生成全景图"""
    global text2pano_demo, image2pano_demo
    
    try:
        # 创建输出目录
        timestamp = int(time.time())
        output_dir = f"gradio_outputs/pano_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        if mode == "Text to World":
            if not prompt.strip():
                return None, "❌ 请输入文本提示词"
            
            # 按需加载模型
            if not load_model_on_demand("text2pano"):
                return None, "❌ 文本到全景图模型加载失败"
            
            print(f"🎨 开始生成文本到全景图: {prompt}")
            result_image = text2pano_demo.run(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                seed=seed,
                output_path=output_dir
            )
            
        else:  # Image to World
            if input_image is None:
                return None, "❌ 请上传输入图像"
            
            # 按需加载模型
            if not load_model_on_demand("image2pano"):
                return None, "❌ 图像到全景图模型加载失败"
            
            # 如果是从示例加载的，需要复制到临时位置
            if isinstance(input_image, str) and input_image.startswith("examples/"):
                temp_image_path = input_image
            else:
                # 保存上传的图像
                temp_image_path = os.path.join(output_dir, "input.png")
                try:
                    if hasattr(input_image, 'save'):
                        # PIL Image
                        input_image.save(temp_image_path)
                    else:
                        # numpy array 或其他格式
                        from PIL import Image
                        if isinstance(input_image, np.ndarray):
                            if input_image.dtype != np.uint8:
                                input_image = (input_image * 255).astype(np.uint8)
                            pil_image = Image.fromarray(input_image)
                            pil_image.save(temp_image_path)
                        else:
                            raise ValueError(f"Unsupported image type: {type(input_image)}")
                except Exception as img_save_error:
                    return None, f"❌ 保存输入图像失败: {str(img_save_error)}"
            
            print(f"🖼️ 开始生成图像到全景图: {temp_image_path}")
            result_image = image2pano_demo.run(
                prompt=prompt if prompt else "",
                negative_prompt=negative_prompt if negative_prompt else "",
                image_path=temp_image_path,
                seed=seed,
                output_path=output_dir
            )
        
        panorama_path = os.path.join(output_dir, "panorama.png")
        if os.path.exists(panorama_path):
            # 强制清理内存
            print("🧹 清理全景图生成后的内存...")
            import gc
            del result_image  # 删除结果图像引用
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保CUDA操作完成
            print(f"📊 内存清理后状态: {get_memory_info()}")
            
            return panorama_path, f"✅ 全景图生成成功！保存到: {output_dir}"
        else:
            return None, "❌ 全景图生成失败"
            
    except Exception as e:
        return None, f"❌ 生成全景图时出错: {str(e)}"

def generate_3d_world(panorama_image, classes, labels_fg1, labels_fg2, seed):
    """生成3D世界"""
    global scenegen_demo
    
    try:
        if panorama_image is None:
            return [], "❌ 请先生成全景图", "<div style='text-align: center; padding: 50px; color: #666;'>请先生成全景图</div>"
        
        # 在开始3D世界生成前清理内存
        print("🧹 开始3D世界生成前的内存清理...")
        clear_unused_models()  # 清理全景图模型
        
        # 按需加载模型
        if not load_model_on_demand("scenegen"):
            return [], "❌ 场景生成模型加载失败", "<div style='text-align: center; padding: 50px; color: #f44336;'>模型加载失败</div>"
        
        # 创建输出目录
        timestamp = int(time.time())
        output_dir = f"gradio_outputs/world_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 复制全景图到输出目录
        panorama_path = os.path.join(output_dir, "panorama.png")
        if isinstance(panorama_image, str):
            shutil.copy2(panorama_image, panorama_path)
        else:
            # 处理不同类型的图像数据
            try:
                if hasattr(panorama_image, 'save'):
                    # PIL Image
                    panorama_image.save(panorama_path)
                else:
                    # numpy array 或其他格式
                    from PIL import Image
                    import numpy as np
                    if isinstance(panorama_image, np.ndarray):
                        # 将numpy数组转换为PIL Image
                        if panorama_image.dtype != np.uint8:
                            panorama_image = (panorama_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(panorama_image)
                        pil_image.save(panorama_path)
                    else:
                        raise ValueError(f"Unsupported image type: {type(panorama_image)}")
            except Exception as img_save_error:
                return [], f"❌ 保存全景图失败: {str(img_save_error)}", "<div style='text-align: center; padding: 50px; color: #f44336;'>图像保存失败</div>"
        
        print(f"🌍 开始生成3D世界场景...")
        
        # 处理标签
        fg1_labels = [label.strip() for label in labels_fg1.split()] if labels_fg1.strip() else []
        fg2_labels = [label.strip() for label in labels_fg2.split()] if labels_fg2.strip() else []
        
        scenegen_demo.run(
            image_path=panorama_path,
            labels_fg1=fg1_labels,
            labels_fg2=fg2_labels,
            classes=classes if classes else "outdoor",
            output_dir=output_dir,
            export_drc=False
        )
        
        # 查找生成的文件
        result_files = []
        for ply_file in Path(output_dir).glob("mesh_layer*.ply"):
            result_files.append(str(ply_file))
        
        if result_files:
            # 生成3D预览页面
            viewer_html_content = create_3d_viewer_content(result_files)
            return result_files, f"✅ 3D世界生成成功！共生成 {len(result_files)} 个网格文件\n保存到: {output_dir}", viewer_html_content
        else:
            return [], "❌ 3D世界生成失败，未找到输出文件", "<div style='text-align: center; padding: 50px; color: #666;'>生成失败</div>"
            
    except Exception as e:
        return [], f"❌ 生成3D世界时出错: {str(e)}", "<div style='text-align: center; padding: 50px; color: #f44336;'>生成失败</div>"

def create_gradio_app():

    """创建Gradio应用界面"""
    
    # 获取示例案例
    cases = get_example_cases()
    case_names = [""] + list(cases.keys())
    
    with gr.Blocks(title="HunyuanWorld 1.0 Web Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🌍 HunyuanWorld 1.0 Web Demo
        
        **欢迎使用 HunyuanWorld 1.0！** 这是一个强大的3D世界生成工具，支持从文本或图像生成沉浸式的3D世界。
        
        ## 使用流程：
        1. **选择示例** 或 **自定义输入**
        2. **生成全景图** - 从文本或图像生成360°全景图
        3. **生成3D世界** - 将全景图转换为可探索的3D场景
        
        ⚠️ **内存优化提示：** 模型将在首次使用时自动加载，以节省内存资源
        """)
        
        # 内存监控区域
        with gr.Row():
            with gr.Column(scale=3):
                memory_display = gr.Textbox(
                    label="系统状态",
                    value=get_memory_info(),
                    interactive=False
                )
            with gr.Column(scale=1):
                refresh_memory_btn = gr.Button("🔄 刷新状态", size="sm")
                clear_cache_btn = gr.Button("🧹 清理缓存", size="sm")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 步骤1: 选择示例或自定义输入")
                
                case_dropdown = gr.Dropdown(
                    choices=case_names,
                    label="选择示例案例",
                    value="",
                    interactive=True
                )
                
                mode_radio = gr.Radio(
                    choices=["Text to World", "Image to World"],
                    label="生成模式",
                    value="Text to World"
                )
                
                with gr.Group():
                    prompt_textbox = gr.Textbox(
                        label="文本提示词",
                        placeholder="输入您想要生成的场景描述...",
                        lines=3
                    )
                    
                    negative_prompt_textbox = gr.Textbox(
                        label="负面提示词（可选）",
                        placeholder="输入不希望出现的内容...",
                        lines=2
                    )
                    
                    input_image = gr.Image(
                        label="输入图像（Image to World 模式）",
                        type="pil"
                    )
                
                seed_number = gr.Number(
                    label="随机种子",
                    value=42,
                    precision=0
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 步骤2: 生成全景图")
                
                generate_pano_btn = gr.Button("🚀 生成全景图", variant="primary", size="lg")
                
                panorama_output = gr.Image(
                    label="生成的全景图",
                    interactive=False
                )
                
                pano_status = gr.Textbox(
                    label="状态信息",
                    interactive=False,
                    lines=2
                )
                
                # 内存提示
                gr.Markdown("""
                💡 **内存优化提示**: 全景图生成完成后，建议点击上方"清理缓存"按钮释放内存，然后再进行3D世界生成。
                """, elem_classes="memory-tip")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🌍 步骤3: 生成3D世界")
                
                with gr.Row():
                    classes_textbox = gr.Textbox(
                        label="场景类别",
                        value="outdoor",
                        placeholder="indoor 或 outdoor"
                    )
                    
                    labels_fg1_textbox = gr.Textbox(
                        label="前景层1标签",
                        placeholder="例如: stones sculptures flowers"
                    )
                    
                    labels_fg2_textbox = gr.Textbox(
                        label="前景层2标签", 
                        placeholder="例如: trees mountains"
                    )
                
                generate_world_btn = gr.Button("🌟 生成3D世界", variant="secondary", size="lg")
                
                world_files = gr.File(
                    label="生成的3D网格文件",
                    file_count="multiple",
                    interactive=False
                )
                
                world_status = gr.Textbox(
                    label="状态信息",
                    interactive=False,
                    lines=3
                )
                
                # 3D预览区域
                gr.Markdown("### 🔍 3D模型预览")
                
                viewer_html = gr.HTML(
                    label="3D世界预览",
                    value="<div style='text-align: center; padding: 50px; color: #666;'>生成3D世界后，这里将显示预览信息和使用说明</div>"
                )
        
        # 事件处理
        def on_case_change(case_name):
            return load_example_case(case_name)
        
        def on_refresh_memory():
            return get_memory_info()
        
        def on_clear_cache():
            clear_unused_models()
            return get_memory_info()
        
        case_dropdown.change(
            fn=on_case_change,
            inputs=[case_dropdown],
            outputs=[input_image, prompt_textbox, classes_textbox, labels_fg1_textbox, labels_fg2_textbox, mode_radio]
        )
        
        refresh_memory_btn.click(
            fn=on_refresh_memory,
            inputs=[],
            outputs=[memory_display]
        )
        
        clear_cache_btn.click(
            fn=on_clear_cache,
            inputs=[],
            outputs=[memory_display]
        )
        
        generate_pano_btn.click(
            fn=generate_panorama,
            inputs=[mode_radio, prompt_textbox, negative_prompt_textbox, input_image, seed_number, case_dropdown],
            outputs=[panorama_output, pano_status]
        )
        
        generate_world_btn.click(
            fn=generate_3d_world,
            inputs=[panorama_output, classes_textbox, labels_fg1_textbox, labels_fg2_textbox, seed_number],
            outputs=[world_files, world_status, viewer_html]
        )
        
        # 示例展示
        gr.Markdown("""
        ### 📚 示例说明
        
        **Text to World 示例:**
        - Case4: 海上燃烧岩石岛屿场景
        - Case7: 冰川崩塌灾难场景
        - Case9: 火山爆发场景
        
        **Image to World 示例:**
        - Case1, Case3, Case5, Case8: 各种户外场景
        - Case2: 石头和树木场景（带多层前景）
        - Case6: 帐篷场景
        
        ### 💡 使用提示
        - 模型将在首次使用时自动加载（约1-3分钟）
        - 全景图生成通常需要1-3分钟
        - 3D世界生成需要额外的5-10分钟
        - 生成的PLY文件可以用3D查看器打开
        - 支持使用提供的 `modelviewer.html` 在浏览器中查看3D场景
        
        ### ⚠️ 内存优化
        - 如果遇到内存不足，请点击"清理缓存"按钮
        - 建议单独使用功能，避免同时生成多个任务
        - 大型模型需要足够的GPU/CPU内存支持
        """)
    
    return app

def main():
    """主函数"""
    print("🌍 HunyuanWorld 1.0 Web Demo 启动中...")
    
    # 创建输出目录
    os.makedirs("gradio_outputs", exist_ok=True)
    
    # 加载模型
    if not load_models():
        print("❌ 模型加载失败，程序退出")
        sys.exit(1)
    
    # 创建并启动应用
    app = create_gradio_app()
    
    print("🚀 启动Web界面...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
