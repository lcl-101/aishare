# Copyright (C) 2025 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import gradio as gr
import torch
import random
import io
import base64
from PIL import Image
import numpy as np

from addit_flux_pipeline import AdditFluxPipeline
from addit_flux_transformer import AdditFluxTransformer2DModel
from addit_scheduler import AdditFlowMatchEulerDiscreteScheduler
from addit_methods import add_object_generated, add_object_real

# Global variables for the pipeline
pipe = None
device = None

def initialize_pipeline():
    """Initialize the AdditFlux pipeline"""
    global pipe, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("正在加载AdditFlux模型...")
    my_transformer = AdditFluxTransformer2DModel.from_pretrained(
        "checkpoints/FLUX.1-dev", 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    )

    pipe = AdditFluxPipeline.from_pretrained(
        "checkpoints/FLUX.1-dev", 
        transformer=my_transformer,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    pipe.scheduler = AdditFlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    print("流水线加载成功！")

def generate_image_from_prompts(
    prompt_source,
    prompt_target, 
    subject_token,
    seed_src,
    seed_obj,
    extended_scale,
    structure_transfer_step,
    blend_steps_str,
    show_attention,
    localization_model
):
    """Generate image using text prompts"""
    global pipe
    
    if pipe is None:
        return None, None, "流水线未初始化。请等待初始化完成。"
    
    try:
        # Parse blend steps
        blend_steps = [int(x.strip()) for x in blend_steps_str.split(',') if x.strip()]
        
        # Reset GPU memory tracking
        torch.cuda.reset_max_memory_allocated(0)
        
        # Generate images
        src_image, edited_image = add_object_generated(
            pipe=pipe,
            prompt_source=prompt_source,
            prompt_object=prompt_target,
            subject_token=subject_token,
            seed_src=seed_src,
            seed_obj=seed_obj,
            show_attention=show_attention,
            extended_scale=extended_scale,
            structure_transfer_step=structure_transfer_step,
            blend_steps=blend_steps,
            localization_model=localization_model,
            display_output=False
        )
        
        # Report GPU memory usage
        max_memory_used = torch.cuda.max_memory_allocated(0) / (1024**3)
        status_msg = f"生成完成！最大GPU内存使用量：{max_memory_used:.2f} GB"
        
        return src_image, edited_image, status_msg
        
    except Exception as e:
        return None, None, f"生成过程中发生错误：{str(e)}"

def generate_from_real_image(
    source_image,
    prompt_source,
    prompt_target,
    subject_token,
    seed_src,
    seed_obj,
    extended_scale,
    structure_transfer_step,
    blend_steps_str,
    show_attention,
    localization_model,
    use_offset,
    use_inversion
):
    """Generate image using real source image"""
    global pipe
    
    if pipe is None:
        return None, None, "流水线未初始化。请等待初始化完成。"
    
    if source_image is None:
        return None, None, "请提供源图像。"
    
    try:
        # Parse blend steps
        blend_steps = [int(x.strip()) for x in blend_steps_str.split(',') if x.strip()]
        
        # Reset GPU memory tracking
        torch.cuda.reset_max_memory_allocated(0)
        
        # Generate images
        src_image, edited_image = add_object_real(
            pipe=pipe,
            source_image=source_image,
            prompt_source=prompt_source,
            prompt_object=prompt_target,
            subject_token=subject_token,
            seed_src=seed_src,
            seed_obj=seed_obj,
            localization_model=localization_model,
            extended_scale=extended_scale,
            structure_transfer_step=structure_transfer_step,
            blend_steps=blend_steps,
            use_offset=use_offset,
            show_attention=show_attention,
            use_inversion=use_inversion,
            display_output=False
        )
        
        # Report GPU memory usage
        max_memory_used = torch.cuda.max_memory_allocated(0) / (1024**3)
        status_msg = f"生成完成！最大GPU内存使用量：{max_memory_used:.2f} GB"
        
        return src_image, edited_image, status_msg
        
    except Exception as e:
        return None, None, f"生成过程中发生错误：{str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="AdditFlux - 图像对象添加", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AdditFlux - 图像对象添加")
        gr.Markdown("使用AI驱动的扩散模型向图像中添加对象。")
        
        with gr.Tabs():
            # Tab 1: Generated Source Image
            with gr.TabItem("生成源图像"):
                gr.Markdown("### 从文本提示生成图像")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_source = gr.Textbox(
                            label="源提示词",
                            placeholder="A photo of a man sitting on a bench",
                            value="A photo of a man sitting on a bench"
                        )
                        prompt_target = gr.Textbox(
                            label="目标提示词（添加对象后）",
                            placeholder="A photo of a man sitting on a bench with a dog",
                            value="A photo of a man sitting on a bench with a dog"
                        )
                        subject_token = gr.Textbox(
                            label="主体词（要添加的对象）",
                            placeholder="dog",
                            value="dog"
                        )
                        
                        with gr.Row():
                            seed_src = gr.Number(
                                label="源随机种子",
                                value=663,
                                precision=0
                            )
                            seed_obj = gr.Number(
                                label="对象随机种子", 
                                value=0,
                                precision=0
                            )
                        
                        with gr.Accordion("高级设置", open=False):
                            extended_scale = gr.Slider(
                                label="扩展比例",
                                minimum=0.5,
                                maximum=2.0,
                                value=1.05,
                                step=0.05
                            )
                            structure_transfer_step = gr.Slider(
                                label="结构传输步骤",
                                minimum=0,
                                maximum=10,
                                value=2,
                                step=1
                            )
                            blend_steps_str = gr.Textbox(
                                label="混合步骤（逗号分隔）",
                                value="15",
                                placeholder="15,20,25"
                            )
                            show_attention = gr.Checkbox(
                                label="显示注意力",
                                value=True
                            )
                            localization_model = gr.Dropdown(
                                label="定位模型",
                                choices=["attention_points_sam", "other"],
                                value="attention_points_sam"
                            )
                        
                        generate_btn = gr.Button("生成图像", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            output_src = gr.Image(label="源图像", type="pil")
                            output_edited = gr.Image(label="编辑后图像", type="pil")
                        
                        status_text = gr.Textbox(label="状态", interactive=False)
                
                generate_btn.click(
                    fn=generate_image_from_prompts,
                    inputs=[
                        prompt_source, prompt_target, subject_token,
                        seed_src, seed_obj, extended_scale, structure_transfer_step,
                        blend_steps_str, show_attention, localization_model
                    ],
                    outputs=[output_src, output_edited, status_text]
                )
            
            # Tab 2: Real Source Image  
            with gr.TabItem("真实源图像"):
                gr.Markdown("### 向现有图像添加对象")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        source_image_input = gr.Image(
                            label="源图像",
                            type="pil"
                        )
                        prompt_source_real = gr.Textbox(
                            label="源图像描述",
                            placeholder="A photo of a room",
                            value="A photo of a room"
                        )
                        prompt_target_real = gr.Textbox(
                            label="目标提示词（添加对象后）",
                            placeholder="A photo of a room with a cat",
                            value="A photo of a room with a cat"
                        )
                        subject_token_real = gr.Textbox(
                            label="主体词（要添加的对象）",
                            placeholder="cat",
                            value="cat"
                        )
                        
                        with gr.Row():
                            seed_src_real = gr.Number(
                                label="源随机种子",
                                value=663,
                                precision=0
                            )
                            seed_obj_real = gr.Number(
                                label="对象随机种子",
                                value=0,
                                precision=0
                            )
                        
                        with gr.Accordion("高级设置", open=False):
                            extended_scale_real = gr.Slider(
                                label="扩展比例",
                                minimum=0.5,
                                maximum=2.0,
                                value=1.05,
                                step=0.05
                            )
                            structure_transfer_step_real = gr.Slider(
                                label="结构传输步骤",
                                minimum=0,
                                maximum=10,
                                value=4,
                                step=1
                            )
                            blend_steps_str_real = gr.Textbox(
                                label="混合步骤（逗号分隔）",
                                value="20",
                                placeholder="15,20,25"
                            )
                            show_attention_real = gr.Checkbox(
                                label="显示注意力",
                                value=False
                            )
                            localization_model_real = gr.Dropdown(
                                label="定位模型",
                                choices=["attention_points_sam", "other"],
                                value="attention_points_sam"
                            )
                            use_offset = gr.Checkbox(
                                label="使用偏移",
                                value=False
                            )
                            use_inversion = gr.Checkbox(
                                label="使用反演",
                                value=False
                            )
                        
                        generate_real_btn = gr.Button("生成图像", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            output_src_real = gr.Image(label="源图像", type="pil")
                            output_edited_real = gr.Image(label="编辑后图像", type="pil")
                        
                        status_text_real = gr.Textbox(label="状态", interactive=False)
                
                generate_real_btn.click(
                    fn=generate_from_real_image,
                    inputs=[
                        source_image_input, prompt_source_real, prompt_target_real, subject_token_real,
                        seed_src_real, seed_obj_real, extended_scale_real, structure_transfer_step_real,
                        blend_steps_str_real, show_attention_real, localization_model_real,
                        use_offset, use_inversion
                    ],
                    outputs=[output_src_real, output_edited_real, status_text_real]
                )
                
                # Real image examples
                with gr.Accordion("真实图像示例", open=False):
                    gr.Examples(
                        examples=[
                            ["images/bed_dark_room.jpg", "A photo of a bed in a dark room", "A photo of a dog lying on a bed in a dark room", "dog"],
                            ["images/cat.jpg", "A photo of a cat", "A photo of a cat with a hat", "hat"],
                        ],
                        inputs=[source_image_input, prompt_source_real, prompt_target_real, subject_token_real]
                    )
        
        # Example gallery
        with gr.Accordion("示例图像", open=False):
            gr.Examples(
                examples=[
                    ["A photo of a man sitting on a bench", "A photo of a man sitting on a bench with a dog", "dog"],
                    ["A bedroom interior", "A bedroom interior with a cat", "cat"],
                    ["An empty park", "An empty park with children playing", "children"],
                ],
                inputs=[prompt_source, prompt_target, subject_token]
            )
    
    return demo

def main():
    """Main function to run the application"""
    # Initialize the pipeline
    initialize_pipeline()
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
