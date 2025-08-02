#!/usr/bin/env python3
"""
FLUX WebUI - A Gradio-based web interface for FLUX image generation
使用本地下载的 FLUX 模型进行图像生成
"""

import torch
import gradio as gr
from diffusers import FluxPipeline
import os
from PIL import Image
import time

class FluxWebUI:
    def __init__(self):
        self.pipe = None
        self.current_model = None
        self.model_paths = {
            "FLUX.1-dev": "checkpoints/FLUX.1-dev",
            "FLUX.1-Krea-dev": "checkpoints/FLUX.1-Krea-dev"
        }
        
    def load_model(self, model_name):
        """加载指定的模型"""
        if self.current_model == model_name and self.pipe is not None:
            return f"模型 {model_name} 已经加载"
            
        try:
            model_path = self.model_paths[model_name]
            print(f"正在加载模型: {model_name} 从路径: {model_path}")
            
            # 释放之前的模型内存
            if self.pipe is not None:
                del self.pipe
                torch.cuda.empty_cache()
            
            # 检查GPU可用性
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {device}")
            if device == "cuda":
                print(f"GPU 名称: {torch.cuda.get_device_name()}")
                print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # 加载新模型并优化GPU使用
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 尝试不同的加载策略
            try:
                # 方法1: 直接加载
                self.pipe = FluxPipeline.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
                self.pipe = self.pipe.to(device)
                print("✅ 使用直接加载方法")
            except Exception as e1:
                print(f"直接加载失败，尝试其他方法: {e1}")
                try:
                    # 方法2: 使用balanced策略
                    self.pipe = FluxPipeline.from_pretrained(
                        model_path, 
                        torch_dtype=torch.bfloat16,
                        local_files_only=True,
                        device_map="balanced",
                    )
                    print("✅ 使用balanced策略")
                except Exception as e2:
                    print(f"balanced策略失败: {e2}")
                    # 方法3: CPU模式
                    self.pipe = FluxPipeline.from_pretrained(
                        model_path, 
                        torch_dtype=torch.bfloat16,
                        local_files_only=True,
                    )
                    device = "cpu"
                    print("⚠️ 回退到CPU模式")
            
            # 不使用CPU offload，因为您有足够的显存
            # self.pipe.enable_model_cpu_offload()  # 注释掉这行
            
            # 启用其他GPU优化
            if device == "cuda":
                # 启用内存高效注意力机制
                self.pipe.enable_attention_slicing()
                # 启用VAE切片以减少内存使用
                self.pipe.enable_vae_slicing()
                # 如果支持，启用flash attention
                try:
                    self.pipe.enable_flash_attention()
                    print("✅ Flash Attention 已启用")
                except:
                    print("⚠️ Flash Attention 不可用")
                
                # 清理GPU内存
                torch.cuda.empty_cache()
                
            self.current_model = model_name
            return f"✅ 成功加载模型: {model_name} 到 {device.upper()}"
            
        except Exception as e:
            return f"❌ 加载模型失败: {str(e)}"
    
    def generate_image(self, prompt, model_name, height, width, guidance_scale, num_inference_steps):
        """生成图像"""
        if self.pipe is None or self.current_model != model_name:
            load_result = self.load_model(model_name)
            if "❌" in load_result:
                return None, load_result
    
    def generate_comparison_images(self, prompt, height, width, guidance_scale, num_inference_steps):
        """同时使用两个模型生成图像进行对比"""
        results = {}
        total_start_time = time.time()
        
        # 获取可用的模型列表
        available_models = [name for name, path in self.model_paths.items() if os.path.exists(path)]
        
        if len(available_models) < 2:
            return None, None, "❌ 需要至少两个可用模型才能进行对比"
        
        print(f"🔄 开始对比模式生成...")
        print(f"提示词: {prompt}")
        print(f"尺寸: {width}x{height}")
        print(f"引导比例: {guidance_scale}")
        print(f"推理步数: {num_inference_steps}")
        
        status_messages = []
        
        for i, model_name in enumerate(available_models):
            try:
                print(f"\n🎨 正在使用模型 {i+1}/2: {model_name}")
                
                # 加载模型
                if self.current_model != model_name:
                    load_result = self.load_model(model_name)
                    if "❌" in load_result:
                        status_messages.append(f"❌ {model_name}: 加载失败")
                        continue
                
                # 检查GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"GPU 内存使用: {memory_allocated:.1f}GB")
                
                start_time = time.time()
                
                # 生成图像
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    result = self.pipe(
                        prompt,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=torch.Generator().manual_seed(42),  # 固定种子确保可重复
                    )
                
                generation_time = time.time() - start_time
                image = result.images[0]
                
                # 保存图像
                timestamp = int(time.time())
                filename = f"flux_compare_{model_name.replace('.', '_')}_{timestamp}.png"
                filepath = f"/workspace/flux/{filename}"
                image.save(filepath)
                
                results[model_name] = {
                    'image': image,
                    'time': generation_time,
                    'filename': filename
                }
                
                status_messages.append(f"✅ {model_name}: {generation_time:.2f}秒 -> {filename}")
                print(f"✅ {model_name} 生成完成: {generation_time:.2f}秒")
                
                # 清理GPU内存
                torch.cuda.empty_cache()
                
            except Exception as e:
                error_msg = f"❌ {model_name} 生成失败: {str(e)}"
                status_messages.append(error_msg)
                print(error_msg)
        
        total_time = time.time() - total_start_time
        
        # 准备返回结果
        model_names = list(results.keys())
        if len(model_names) >= 2:
            image1 = results[model_names[0]]['image']
            image2 = results[model_names[1]]['image']
            
            final_status = f"🎉 对比生成完成！总耗时: {total_time:.2f}秒\n\n"
            final_status += "\n".join(status_messages)
            
            # 显示最终GPU内存使用情况
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**3
                final_status += f"\n\n💾 最终GPU内存使用: {final_memory:.1f}GB"
            
            return image1, image2, final_status
        elif len(model_names) == 1:
            model_name = model_names[0]
            return results[model_name]['image'], None, f"⚠️ 只有一个模型生成成功: {model_name}\n" + "\n".join(status_messages)
        else:
            return None, None, "❌ 所有模型都生成失败\n" + "\n".join(status_messages)

    def generate_single_image(self, prompt, model_name, height, width, guidance_scale, num_inference_steps):
        """使用单个模型生成图像"""
        if self.pipe is None or self.current_model != model_name:
            load_result = self.load_model(model_name)
            if "❌" in load_result:
                return None, load_result
        
        try:
            print(f"正在生成图像...")
            print(f"提示词: {prompt}")
            print(f"尺寸: {width}x{height}")
            print(f"引导比例: {guidance_scale}")
            print(f"推理步数: {num_inference_steps}")
            
            # 检查GPU内存使用情况
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理内存
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU 内存使用: {memory_allocated:.1f}GB 已分配, {memory_reserved:.1f}GB 已保留")
            
            start_time = time.time()
            
            # 生成图像，确保使用GPU
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                result = self.pipe(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator().manual_seed(42),  # 固定种子以便复现
                )
            
            image = result.images[0]
            generation_time = time.time() - start_time
            
            # 显示最终GPU内存使用情况
            if torch.cuda.is_available():
                final_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                final_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"生成后GPU内存: {final_memory_allocated:.1f}GB 已分配, {final_memory_reserved:.1f}GB 已保留")
            
            # 保存图像
            timestamp = int(time.time())
            filename = f"flux_generated_{timestamp}.png"
            filepath = f"/workspace/flux/{filename}"
            image.save(filepath)
            
            status = f"✅ 图像生成成功！\n时间: {generation_time:.2f}秒\n保存路径: {filename}\nGPU内存使用: {final_memory_allocated:.1f}GB"
            return image, status
            
        except Exception as e:
            error_msg = f"❌ 图像生成失败: {str(e)}"
            print(error_msg)
            return None, error_msg
        
        try:
            print(f"正在生成图像...")
            print(f"提示词: {prompt}")
            print(f"尺寸: {width}x{height}")
            print(f"引导比例: {guidance_scale}")
            print(f"推理步数: {num_inference_steps}")
            
            # 检查GPU内存使用情况
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理内存
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU 内存使用: {memory_allocated:.1f}GB 已分配, {memory_reserved:.1f}GB 已保留")
            
            start_time = time.time()
            
            # 生成图像，确保使用GPU
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                result = self.pipe(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator().manual_seed(42),  # 固定种子以便复现
                )
            
            image = result.images[0]
            generation_time = time.time() - start_time
            
            # 显示最终GPU内存使用情况
            if torch.cuda.is_available():
                final_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                final_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"生成后GPU内存: {final_memory_allocated:.1f}GB 已分配, {final_memory_reserved:.1f}GB 已保留")
            
            # 保存图像
            timestamp = int(time.time())
            filename = f"flux_generated_{timestamp}.png"
            filepath = f"/workspace/flux/{filename}"
            image.save(filepath)
            
            status = f"✅ 图像生成成功！\n时间: {generation_time:.2f}秒\n保存路径: {filename}\nGPU内存使用: {final_memory_allocated:.1f}GB"
            return image, status
            
        except Exception as e:
            error_msg = f"❌ 图像生成失败: {str(e)}"
            print(error_msg)
            return None, error_msg

def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return "❌ CUDA 不可用"
    
    gpu_name = torch.cuda.get_device_name()
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_free = memory_total - memory_allocated
    
    return f"""
🔧 **GPU状态:**
- 设备: {gpu_name}  
- 总内存: {memory_total:.1f} GB
- 已使用: {memory_allocated:.1f} GB  
- 可用: {memory_free:.1f} GB
"""

def create_interface():
    """创建 Gradio 界面"""
    flux_ui = FluxWebUI()
    
    with gr.Blocks(title="FLUX Image Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 FLUX Image Generator WebUI
        
        基于 FLUX 模型的图像生成工具，使用本地预下载的模型文件。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 生成模式选择
                generation_mode = gr.Radio(
                    choices=["单模型生成", "双模型对比"],
                    value="单模型生成",
                    label="生成模式",
                    info="选择单个模型生成或两个模型对比生成"
                )
                
                # 模型选择（只在单模型模式下显示）
                model_dropdown = gr.Dropdown(
                    choices=list(flux_ui.model_paths.keys()),
                    value="FLUX.1-Krea-dev",
                    label="选择模型",
                    info="选择要使用的 FLUX 模型",
                    visible=True
                )
                
                # 提示词输入
                prompt_input = gr.Textbox(
                    label="提示词 (Prompt)",
                    placeholder="A frog holding a sign that says hello world",
                    value="A frog holding a sign that says hello world",
                    lines=3
                )
                
                # 生成参数
                with gr.Group():
                    gr.Markdown("### 生成参数")
                    
                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=512, maximum=2048, step=64, value=1024,
                            label="宽度"
                        )
                        height_slider = gr.Slider(
                            minimum=512, maximum=2048, step=64, value=1024,
                            label="高度"
                        )
                    
                    guidance_scale_slider = gr.Slider(
                        minimum=1.0, maximum=20.0, step=0.5, value=4.5,
                        label="引导比例 (Guidance Scale)",
                        info="控制生成图像与提示词的一致性"
                    )
                    
                    steps_slider = gr.Slider(
                        minimum=1, maximum=100, step=1, value=20,
                        label="推理步数 (Inference Steps)",
                        info="更多步数通常产生更好的质量，但需要更长时间"
                    )
                
                # 生成按钮
                with gr.Row():
                    generate_single_btn = gr.Button(
                        "🎨 单模型生成", 
                        variant="primary",
                        size="lg",
                        visible=True
                    )
                    generate_compare_btn = gr.Button(
                        "🔄 双模型对比", 
                        variant="secondary",
                        size="lg",
                        visible=False
                    )
            
            with gr.Column(scale=1):
                # 单模型输出
                with gr.Group(visible=True) as single_output:
                    gr.Markdown("### 生成结果")
                    output_image = gr.Image(
                        label="生成的图像",
                        type="pil",
                        height=600
                    )
                
                # 对比模式输出
                with gr.Group(visible=False) as compare_output:
                    gr.Markdown("### 模型对比结果")
                    with gr.Row():
                        output_image1 = gr.Image(
                            label="FLUX.1-dev",
                            type="pil",
                            height=400
                        )
                        output_image2 = gr.Image(
                            label="FLUX.1-Krea-dev",
                            type="pil",
                            height=400
                        )
                
                # 状态信息
                status_text = gr.Textbox(
                    label="状态信息",
                    lines=5,
                    interactive=False
                )
                
                # GPU状态显示
                gpu_info_text = gr.Markdown(
                    value=get_gpu_info(),
                    label="GPU状态"
                )
                
                # 刷新GPU状态按钮
                refresh_gpu_btn = gr.Button(
                    "🔄 刷新GPU状态",
                    size="sm"
                )
        
        # 示例提示词
        gr.Markdown("### 💡 示例提示词")
        example_prompts = [
            "A majestic dragon flying over a medieval castle at sunset",
            "A futuristic city with flying cars and neon lights",
            "A peaceful zen garden with cherry blossoms",
            "A cute robot playing with a kitten in a cozy room",
            "An astronaut riding a horse on the moon"
        ]
        
        gr.Examples(
            examples=[[prompt] for prompt in example_prompts],
            inputs=[prompt_input],
            label="点击使用示例提示词"
        )
        
        # 模式切换功能
        def toggle_generation_mode(mode):
            if mode == "单模型生成":
                return (
                    gr.update(visible=True),   # model_dropdown
                    gr.update(visible=True),   # generate_single_btn
                    gr.update(visible=False),  # generate_compare_btn
                    gr.update(visible=True),   # single_output
                    gr.update(visible=False),  # compare_output
                )
            else:  # 双模型对比
                return (
                    gr.update(visible=False),  # model_dropdown
                    gr.update(visible=False),  # generate_single_btn
                    gr.update(visible=True),   # generate_compare_btn
                    gr.update(visible=False),  # single_output
                    gr.update(visible=True),   # compare_output
                )
        
        generation_mode.change(
            fn=toggle_generation_mode,
            inputs=[generation_mode],
            outputs=[model_dropdown, generate_single_btn, generate_compare_btn, single_output, compare_output]
        )
        
        # 绑定单模型生成事件
        generate_single_btn.click(
            fn=flux_ui.generate_single_image,
            inputs=[
                prompt_input,
                model_dropdown,
                height_slider,
                width_slider,
                guidance_scale_slider,
                steps_slider
            ],
            outputs=[output_image, status_text]
        )
        
        # 绑定双模型对比生成事件
        generate_compare_btn.click(
            fn=flux_ui.generate_comparison_images,
            inputs=[
                prompt_input,
                height_slider,
                width_slider,
                guidance_scale_slider,
                steps_slider
            ],
            outputs=[output_image1, output_image2, status_text]
        )
        
        # 绑定GPU状态刷新事件
        refresh_gpu_btn.click(
            fn=get_gpu_info,
            outputs=[gpu_info_text]
        )
    
    return demo

def main():
    """主函数"""
    print("🚀 启动 FLUX WebUI...")
    
    # 检查模型文件是否存在
    model_paths = {
        "FLUX.1-dev": "checkpoints/FLUX.1-dev",
        "FLUX.1-Krea-dev": "checkpoints/FLUX.1-Krea-dev"
    }
    
    available_models = []
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            available_models.append(model_name)
            print(f"✅ 发现模型: {model_name}")
        else:
            print(f"❌ 模型不存在: {model_name} (路径: {model_path})")
    
    if not available_models:
        print("❌ 未找到任何可用的模型文件！")
        return
    
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"🔍 可用模型: {', '.join(available_models)}")
    
    # 创建并启动界面
    demo = create_interface()
    
    # 启动服务器
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口
        share=False,            # 不创建公共链接
        debug=True,             # 调试模式
        show_error=True         # 显示错误信息
    )

if __name__ == "__main__":
    main()
