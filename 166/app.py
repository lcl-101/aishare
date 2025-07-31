#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniPic Gradio Web UI
A comprehensive web interface for image generation, editing, and understanding tasks.
"""

import os
import sys
import json
import numpy as np
import torch
import gradio as gr
from PIL import Image
from einops import rearrange
from mmengine.config import Config
from src.builder import BUILDER
from src.datasets.utils import crop2square
import tempfile
from datetime import datetime

# Set PYTHONPATH
if "./" not in sys.path:
    sys.path.append("./")

# Configuration paths
CONFIG_PATH = "configs/models/qwen2_5_1_5b_kl16_mar_h.py"
CHECKPOINT_PATH = "checkpoint/pytorch_model.bin"
DEFAULT_IMAGE_SIZE = 1024


class UniPicApp:
    """UniPic application wrapper for all tasks."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_token_idx = None
        self.cfg = None
        self.load_model()
    
    def load_model(self):
        """Load the UniPic model."""
        print("Loading UniPic model...")
        
        # Load configuration
        self.cfg = Config.fromfile(CONFIG_PATH)
        
        # Build and load model
        self.model = BUILDER.build(self.cfg.model).eval().cuda().to(torch.bfloat16)
        
        # Load checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")
        
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': ["<image>"]}
        self.model.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_idx = self.model.tokenizer.encode("<image>", add_special_tokens=False)[-1]
        
        print("Model loaded successfully!")
    
    def preprocess_image_with_dtype(self, image: Image.Image, image_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Process PIL image to normalized tensor [1,C,H,W] - matches original implementation."""
        img = crop2square(image)
        img = img.resize((image_size, image_size))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = 2 * arr - 1
        tensor = torch.from_numpy(arr).to(dtype=dtype)
        return rearrange(tensor, "h w c -> 1 c h w")
    
    def preprocess_image(self, image: Image.Image, image_size: int = None) -> torch.Tensor:
        """Preprocess PIL image to normalized tensor."""
        if image_size is None:
            image_size = DEFAULT_IMAGE_SIZE
            
        img = crop2square(image)
        img = img.resize((image_size, image_size))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = 2 * arr - 1
        tensor = torch.from_numpy(arr).to(dtype=self.model.dtype)
        return rearrange(tensor, "h w c -> 1 c h w")
    
    def expand2square(self, pil_img, background_color=(127, 127, 127)):
        """Expand image to square."""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    def text_to_image(self, prompt, cfg=3.0, temperature=1.0, num_iter=50, 
                     grid_size=2, image_size=1024, cfg_prompt="Generate an image."):
        """Generate image from text prompt."""
        try:
            # Prepare prompt
            full_prompt = f"Generate an image: {prompt}"
            
            # Prepare text conditions
            class_info = self.model.prepare_text_conditions(full_prompt, cfg_prompt)
            input_ids = class_info['input_ids']
            attention_mask = class_info['attention_mask']
            
            # Handle CFG
            if cfg == 1.0:
                input_ids = input_ids[:1]
                attention_mask = attention_mask[:1]
            
            # Repeat for batch
            bsz = grid_size ** 2
            if cfg != 1.0:
                input_ids = torch.cat([
                    input_ids[:1].expand(bsz, -1),
                    input_ids[1:].expand(bsz, -1),
                ])
                attention_mask = torch.cat([
                    attention_mask[:1].expand(bsz, -1),
                    attention_mask[1:].expand(bsz, -1),
                ])
            else:
                input_ids = input_ids.expand(bsz, -1)
                attention_mask = attention_mask.expand(bsz, -1)
            
            # Sample
            m = n = image_size // 16
            
            with torch.no_grad():
                samples = self.model.sample(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    num_iter=num_iter, 
                    cfg=cfg, 
                    cfg_schedule="constant",
                    temperature=temperature, 
                    progress=True, 
                    image_shape=(m, n)
                )
            
            # Convert to image
            samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=grid_size, n=grid_size)
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            
            result_image = Image.fromarray(samples)
            return result_image, "图像生成成功！"
            
        except Exception as e:
            return None, f"图像生成出错: {str(e)}"
    
    def image_to_text(self, image, prompt="Describe the image in detail.", image_size=1024):
        """Generate text description from image."""
        try:
            if image is None:
                return "请先上传图片。"
            
            # Preprocess image
            image = image.convert('RGB')
            image = self.expand2square(image)
            image = image.resize(size=(image_size, image_size))
            image_tensor = torch.from_numpy(np.array(image)).to(dtype=self.model.dtype, device=self.model.device)
            image_tensor = rearrange(image_tensor, 'h w c -> c h w')[None]
            image_tensor = 2 * (image_tensor / 255) - 1
            
            # Prepare prompt
            if hasattr(self.cfg.model, 'prompt_template'):
                full_prompt = self.cfg.model.prompt_template['INSTRUCTION'].format(
                    input="<image>\n" + prompt
                )
            else:
                full_prompt = f"<image>\n{prompt}"
            
            # Replace image token
            image_length = (image_size // 16) ** 2 + 64
            full_prompt = full_prompt.replace('<image>', '<image>' * image_length)
            
            # Tokenize
            input_ids = self.model.tokenizer.encode(
                full_prompt, add_special_tokens=True, return_tensors='pt'
            ).cuda()
            
            # Extract visual features
            with torch.no_grad():
                _, z_enc = self.model.extract_visual_feature(self.model.encode(image_tensor))
            
            # Prepare embeddings
            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.model.llm.config.hidden_size)
            inputs_embeds[input_ids == self.image_token_idx] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != self.image_token_idx] = self.model.llm.get_input_embeddings()(
                input_ids[input_ids != self.image_token_idx]
            )
            
            # Generate
            with torch.no_grad():
                output = self.model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    do_sample=False,
                    max_new_tokens=1024,
                    eos_token_id=self.model.tokenizer.eos_token_id,
                    pad_token_id=self.model.tokenizer.pad_token_id 
                    if self.model.tokenizer.pad_token_id is not None 
                    else self.model.tokenizer.eos_token_id
                )
            
            result = self.model.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if full_prompt.replace('<image>' * image_length, '') in result:
                result = result.split(full_prompt.replace('<image>' * image_length, ''))[-1].strip()
            
            return result
            
        except Exception as e:
            return f"图像处理出错: {str(e)}"
    
    def edit_image(self, source_image, prompt, num_iter=50, cfg=3.0, 
                  cfg_prompt="Repeat this image.", temperature=0.85, image_size=1024):
        """Edit image based on text prompt - follows original EditInferencer implementation exactly."""
        try:
            if source_image is None:
                return None, "请先上传图片。"
            
            grid_size = 1  # Force single image for web UI
            
            # 1) Preprocess source image - exactly like original
            img_tensor = self.preprocess_image_with_dtype(
                source_image, 
                image_size,
                dtype=self.model.dtype
            ).to(self.model.device)
            
            # 2) Encode image and extract features
            with torch.no_grad():
                x_enc = self.model.encode(img_tensor)
                x_con, z_enc = self.model.extract_visual_feature(x_enc)
            
            # 3) Prepare text prompts
            m = n = image_size // 16
            image_length = m * n + 64
            
            if hasattr(self.cfg.model, 'prompt_template'):
                prompt_str = self.cfg.model.prompt_template['INSTRUCTION'].format(
                    input="<image>\n" + prompt.strip()
                )
                cfg_prompt_str = self.cfg.model.prompt_template['INSTRUCTION'].format(
                    input="<image>\n" + cfg_prompt.strip()
                )
            else:
                prompt_str = f"<image>\n{prompt.strip()}"
                cfg_prompt_str = f"<image>\n{cfg_prompt.strip()}"
            
            # Replace <image> token with multiple tokens
            prompt_str = prompt_str.replace('<image>', '<image>' * image_length)
            cfg_prompt_str = cfg_prompt_str.replace('<image>', '<image>' * image_length)
            
            # 4) Tokenize and prepare inputs
            input_ids = self.model.tokenizer.encode(
                prompt_str, add_special_tokens=True, return_tensors='pt')[0].cuda()
            
            if cfg != 1.0:
                null_input_ids = self.model.tokenizer.encode(
                    cfg_prompt_str, add_special_tokens=True, return_tensors='pt')[0].cuda()
                
                # Import pad_sequence locally to match original
                from torch.nn.utils.rnn import pad_sequence
                attention_mask = pad_sequence(
                    [torch.ones_like(input_ids), torch.ones_like(null_input_ids)],
                    batch_first=True, padding_value=0
                ).to(torch.bool)
                input_ids = pad_sequence(
                    [input_ids, null_input_ids],
                    batch_first=True, padding_value=self.model.tokenizer.eos_token_id
                )
            else:
                input_ids = input_ids[None]
                attention_mask = torch.ones_like(input_ids).to(torch.bool)
            
            # 5) Prepare embeddings
            if cfg != 1.0:
                z_enc = torch.cat([z_enc, z_enc], dim=0)
                x_con = torch.cat([x_con, x_con], dim=0)
            
            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.model.llm.config.hidden_size)
            inputs_embeds[input_ids == self.image_token_idx] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != self.image_token_idx] = self.model.llm.get_input_embeddings()(
                input_ids[input_ids != self.image_token_idx]
            )
            
            # 6) Repeat for grid sampling
            bsz = grid_size ** 2
            x_con = torch.cat([x_con] * bsz)
            if cfg != 1.0:
                inputs_embeds = torch.cat([
                    inputs_embeds[:1].expand(bsz, -1, -1),
                    inputs_embeds[1:].expand(bsz, -1, -1),
                ])
                attention_mask = torch.cat([
                    attention_mask[:1].expand(bsz, -1),
                    attention_mask[1:].expand(bsz, -1),
                ])
            else:
                inputs_embeds = inputs_embeds.expand(bsz, -1, -1)
                attention_mask = attention_mask.expand(bsz, -1)
            
            # 7) Sampling - exactly like original
            samples = self.model.sample(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_iter=num_iter,
                cfg=cfg,
                cfg_schedule="constant",  # Use constant instead of cfg_schedule parameter
                temperature=temperature,
                progress=False,  # Set to False for web UI
                image_shape=(m, n),
                x_con=x_con
            )
            
            # 8) Convert to PIL Image - exactly like original
            samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=grid_size, n=grid_size)
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
            out = samples.to("cpu", torch.uint8).numpy()
            result_image = Image.fromarray(out)
            
            return result_image, "图像编辑成功！"
            
        except Exception as e:
            import traceback
            error_msg = f"Error editing image: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # For debugging
            return None, f"图像编辑出错: {str(e)}"


# Initialize the app (will be done in main)
app = None

# Define Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    global app  # Use the global app instance
    
    # Example prompts for text-to-image
    t2i_examples = [
        "A glossy-coated golden retriever stands on the park lawn beside a life-sized penguin statue.",
        "Digital portrait of a girl with rainbow hair.",
        "A majestic mountain landscape with snow-capped peaks and a crystal clear lake.",
        "A futuristic cityscape at night with flying cars and neon lights.",
        "A beautiful butterfly garden with colorful flowers and gentle sunlight.",
        "An ancient castle on a hill surrounded by misty forests."
    ]
    
    # Example prompts for image editing
    edit_examples = [
        "Replace the stars with the candle.",
        "Change the background to a beach scene.",
        "Add some flowers in the foreground.",
        "Make the sky more dramatic with storm clouds.",
        "Replace the grass with snow.",
        "Add a rainbow in the sky."
    ]
    
    # Example prompts for image-to-text
    i2t_examples = [
        "Describe the image in detail.",
        "What objects can you see in this image?",
        "What is the mood or atmosphere of this image?",
        "Describe the colors and lighting in this image.",
        "What activities are happening in this image?",
        "Describe the setting and environment."
    ]
    
    with gr.Blocks(title="UniPic 网页界面", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 UniPic 网页界面")
        gr.Markdown("用于图像生成、编辑和理解任务的综合界面")
        gr.Markdown("💡 **提示**: 点击示例按钮快速开始，或者上传你自己的图片和提示词。")
        
        with gr.Tabs():
            # Text-to-Image Tab
            with gr.Tab("🖼️ 文本生成图像"):
                gr.Markdown("### 根据文本提示生成图像")
                gr.Markdown("输入文本描述，用AI生成精美图像。")
                
                with gr.Row():
                    with gr.Column():
                        t2i_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="在这里输入您的图像描述...",
                            lines=3,
                            value=t2i_examples[0]
                        )
                        
                        # Example buttons for quick prompts
                        with gr.Row():
                            gr.Markdown("**快速示例:**")
                        
                        example_buttons_t2i = []
                        for i in range(0, len(t2i_examples), 3):
                            with gr.Row():
                                for j in range(3):
                                    if i + j < len(t2i_examples):
                                        btn = gr.Button(f"示例 {i+j+1}", size="sm")
                                        example_buttons_t2i.append((btn, t2i_examples[i+j]))
                        
                        with gr.Row():
                            t2i_cfg = gr.Slider(1.0, 10.0, value=3.0, label="CFG 控制强度")
                            t2i_temperature = gr.Slider(0.1, 2.0, value=1.0, label="随机性")
                        
                        with gr.Row():
                            t2i_num_iter = gr.Slider(10, 100, value=50, step=1, label="迭代次数")
                            t2i_grid_size = gr.Slider(1, 4, value=2, step=1, label="生成数量")
                        
                        t2i_image_size = gr.Dropdown(
                            choices=[512, 768, 1024], 
                            value=1024, 
                            label="图像尺寸"
                        )
                        
                        t2i_generate_btn = gr.Button("🚀 生成图像", variant="primary", size="lg")
                    
                    with gr.Column():
                        t2i_output = gr.Image(label="生成的图像", type="pil")
                        t2i_status = gr.Textbox(label="状态", interactive=False)
                
                # Connect example buttons
                for btn, example_text in example_buttons_t2i:
                    btn.click(fn=lambda x=example_text: x, outputs=t2i_prompt)
                
                t2i_generate_btn.click(
                    fn=app.text_to_image,
                    inputs=[t2i_prompt, t2i_cfg, t2i_temperature, t2i_num_iter, 
                           t2i_grid_size, t2i_image_size],
                    outputs=[t2i_output, t2i_status]
                )
            
            # Image Editing Tab
            with gr.Tab("✂️ 图像编辑"):
                gr.Markdown("### 根据文本提示编辑图像")
                gr.Markdown("上传图像并描述您想要的修改。")
                
                with gr.Row():
                    with gr.Column():
                        edit_image_input = gr.Image(label="源图像", type="pil")
                        
                        # Sample image button
                        sample_btn = gr.Button("📁 加载示例图像", size="sm")
                        
                        edit_prompt = gr.Textbox(
                            label="编辑提示词",
                            placeholder="描述您想要如何编辑图像...",
                            lines=2,
                            value=edit_examples[0]
                        )
                        
                        # Example buttons for editing prompts
                        with gr.Row():
                            gr.Markdown("**快速示例:**")
                        
                        example_buttons_edit = []
                        for i in range(0, len(edit_examples), 2):
                            with gr.Row():
                                for j in range(2):
                                    if i + j < len(edit_examples):
                                        btn = gr.Button(f"编辑 {i+j+1}", size="sm")
                                        example_buttons_edit.append((btn, edit_examples[i+j]))
                        
                        with gr.Row():
                            edit_cfg = gr.Slider(1.0, 10.0, value=3.0, label="CFG 控制强度")
                            edit_temperature = gr.Slider(0.1, 2.0, value=0.85, label="随机性")
                        
                        edit_num_iter = gr.Slider(10, 100, value=50, step=1, label="迭代次数")
                        edit_image_size = gr.Dropdown(
                            choices=[512, 768, 1024], 
                            value=1024, 
                            label="图像尺寸"
                        )
                        
                        edit_btn = gr.Button("✨ 编辑图像", variant="primary", size="lg")
                    
                    with gr.Column():
                        edit_output = gr.Image(label="编辑后的图像", type="pil")
                        edit_status = gr.Textbox(label="状态", interactive=False)
                
                # Load sample image function
                def load_sample_image():
                    try:
                        from PIL import Image
                        return Image.open("data/sample.png")
                    except:
                        return None
                
                sample_btn.click(fn=load_sample_image, outputs=edit_image_input)
                
                # Connect example buttons
                for btn, example_text in example_buttons_edit:
                    btn.click(fn=lambda x=example_text: x, outputs=edit_prompt)
                
                edit_btn.click(
                    fn=app.edit_image,
                    inputs=[edit_image_input, edit_prompt, edit_num_iter, edit_cfg, 
                           gr.Textbox(value="Repeat this image.", visible=False), 
                           edit_temperature, edit_image_size],
                    outputs=[edit_output, edit_status]
                )
            
            # Image-to-Text Tab
            with gr.Tab("📖 图像理解"):
                gr.Markdown("### 从图像生成文本描述")
                gr.Markdown("上传图像并询问相关问题或获取详细描述。")
                
                with gr.Row():
                    with gr.Column():
                        i2t_image_input = gr.Image(label="输入图像", type="pil")
                        
                        # Sample image button for i2t
                        sample_btn_i2t = gr.Button("📁 加载示例图像", size="sm")
                        
                        i2t_prompt = gr.Textbox(
                            label="问题",
                            value=i2t_examples[0],
                            lines=2
                        )
                        
                        # Example buttons for i2t prompts
                        with gr.Row():
                            gr.Markdown("**快速示例:**")
                        
                        example_buttons_i2t = []
                        for i in range(0, len(i2t_examples), 2):
                            with gr.Row():
                                for j in range(2):
                                    if i + j < len(i2t_examples):
                                        btn = gr.Button(f"问题 {i+j+1}", size="sm")
                                        example_buttons_i2t.append((btn, i2t_examples[i+j]))
                        
                        i2t_image_size = gr.Dropdown(
                            choices=[512, 768, 1024], 
                            value=1024, 
                            label="图像尺寸"
                        )
                        i2t_btn = gr.Button("🔍 生成描述", variant="primary", size="lg")
                    
                    with gr.Column():
                        i2t_output = gr.Textbox(label="生成的描述", lines=10)
                
                sample_btn_i2t.click(fn=load_sample_image, outputs=i2t_image_input)
                
                # Connect example buttons
                for btn, example_text in example_buttons_i2t:
                    btn.click(fn=lambda x=example_text: x, outputs=i2t_prompt)
                
                i2t_btn.click(
                    fn=app.image_to_text,
                    inputs=[i2t_image_input, i2t_prompt, i2t_image_size],
                    outputs=i2t_output
                )
            
            # Batch Processing Tab
            with gr.Tab("🔄 批量处理"):
                gr.Markdown("### 批量文本生成图像")
                gr.Markdown("从提示词列表生成多张图像。")
                
                with gr.Row():
                    with gr.Column():
                        batch_prompts = gr.Textbox(
                            label="提示词 (每行一个)",
                            placeholder="输入多个提示词，每行一个...",
                            lines=5,
                            value="\n".join(t2i_examples[:4])
                        )
                        
                        # Load example prompts button
                        load_examples_btn = gr.Button("📝 加载示例提示词", size="sm")
                        
                        with gr.Row():
                            batch_cfg = gr.Slider(1.0, 10.0, value=3.0, label="CFG 控制强度")
                            batch_temperature = gr.Slider(0.1, 2.0, value=1.0, label="随机性")
                        
                        batch_num_iter = gr.Slider(10, 100, value=50, step=1, label="迭代次数")
                        batch_image_size = gr.Dropdown(
                            choices=[512, 768, 1024], 
                            value=1024, 
                            label="图像尺寸"
                        )
                        
                        batch_btn = gr.Button("🚀 批量生成", variant="primary", size="lg")
                    
                    with gr.Column():
                        batch_gallery = gr.Gallery(label="生成的图像", columns=2, rows=2)
                        batch_status = gr.Textbox(label="状态", interactive=False)
                
                def load_example_prompts():
                    return "\n".join(t2i_examples)
                
                load_examples_btn.click(fn=load_example_prompts, outputs=batch_prompts)
                
                def batch_text_to_image(prompts_text, cfg, temperature, num_iter, image_size):
                    """Generate multiple images from prompts."""
                    try:
                        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                        if not prompts:
                            return [], "请至少输入一个提示词。"
                        
                        images = []
                        for i, prompt in enumerate(prompts):
                            image, status = app.text_to_image(
                                prompt, cfg, temperature, num_iter, 1, image_size
                            )
                            if image:
                                images.append(image)
                        
                        return images, f"成功生成 {len(images)} 张图像！"
                    except Exception as e:
                        return [], f"批量处理出错: {str(e)}"
                
                batch_btn.click(
                    fn=batch_text_to_image,
                    inputs=[batch_prompts, batch_cfg, batch_temperature, batch_num_iter, batch_image_size],
                    outputs=[batch_gallery, batch_status]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("### 📚 使用说明")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                **文本生成图像:**
                - 使用描述性提示词获得更好效果
                - 更高的CFG值 = 更严格遵循提示词
                - 更多迭代次数 = 更高质量 (但更慢)
                """)
            with gr.Column():
                gr.Markdown("""
                **图像编辑:**
                - 上传清晰的源图像
                - 明确描述所需的修改
                - 尝试不同的随机性值获得多样性
                """)
            with gr.Column():
                gr.Markdown("""
                **图像理解:**
                - 问具体问题获得针对性回答
                - 更高分辨率 = 检测更多细节
                - 尝试不同的问题方式
                """)
        
        gr.Markdown("---")
        gr.Markdown("基于 SkyworkAI 的 UniPic 构建 | 由 Gradio 提供支持")
    
    return interface


if __name__ == "__main__":
    print("🚀 启动 UniPic Gradio 网页界面...")
    
    # Check if gradio is available
    try:
        import gradio as gr
        print("✅ Gradio 导入成功")
    except ImportError:
        print("❌ 未找到 Gradio，正在安装...")
        import os
        os.system("pip install gradio>=4.0.0")
        import gradio as gr
        print("✅ Gradio 安装并导入成功")
    
    # Initialize the app (this will load the model)
    print("🔄 初始化 UniPic 模型...")
    app = UniPicApp()
    print("✅ UniPic 模型加载成功")
    
    # Create and launch the interface
    print("🎨 创建 Gradio 界面...")
    interface = create_interface()
    print("✅ 界面创建成功")
    
    print("")
    print("🌟 UniPic 网页界面已就绪！")
    print("📱 在浏览器中打开: http://localhost:7860")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("")
    print("📋 可用功能:")
    print("   🖼️  文本生成图像")
    print("   ✂️  图像编辑")
    print("   📖 图像理解")
    print("   🔄 批量处理")
    print("")
    
    # Launch with optimal settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        quiet=False,
        show_error=True,
        favicon_path=None,
        ssl_verify=False
    )
