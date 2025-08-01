import os
import torch
import gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import tempfile
import random


class XOmniWebUI:
    def __init__(self):
        self.flux_path = "checkpoints/FLUX.1-dev"
        self.en_model_path = "checkpoints/X-Omni-En"
        self.zh_model_path = "checkpoints/X-Omni-Zh"
        
        # Initialize models
        self.torch_dtype = torch.bfloat16
        self.models = {}
        self.tokenizers = {}
        
        # Load models on demand to save memory
        print("Initializing XOmni WebUI...")
        
    def load_model(self, model_path, model_key):
        """Load model and tokenizer on demand"""
        if model_key not in self.models:
            print(f"Loading {model_key} model...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Move to GPU if not already there
            if not next(model.parameters()).is_cuda:
                model = model.cuda()
                
            model.init_vision(self.flux_path)
            model.eval()
            
            self.tokenizers[model_key] = tokenizer
            self.models[model_key] = model
            print(f"{model_key} model loaded successfully!")
        
        return self.models[model_key], self.tokenizers[model_key]
    
    def generate_image_en(self, prompt, image_size, cfg_scale, min_p, seed, progress=gr.Progress()):
        """Generate image using English model"""
        try:
            print(f"Starting image generation with prompt: {prompt[:50]}...")
            progress(0.1, desc="Loading English model...")
            model, tokenizer = self.load_model(self.en_model_path, "en")
            model.set_generation_mode('image')
            
            progress(0.3, desc="Processing prompt...")
            
            # Parse image size
            if isinstance(image_size, str):
                size_parts = image_size.split('x')
                if len(size_parts) == 2:
                    width, height = int(size_parts[0]), int(size_parts[1])
                else:
                    width = height = int(image_size)
            else:
                width = height = image_size
            
            image_size_tuple = (width, height)
            downsample_size = 16
            
            token_h, token_w = image_size_tuple[0] // downsample_size, image_size_tuple[1] // downsample_size
            image_prefix = f'<SOM>{token_h} {token_w}<IMAGE>'
            
            print(f"Image size: {image_size_tuple}, tokens: {token_h}x{token_w}")
            progress(0.5, desc="Generating image...")
            
            generation_config = GenerationConfig(
                max_new_tokens=token_h * token_w,
                do_sample=True,
                temperature=1.0,
                min_p=min_p,
                top_p=1.0,
                guidance_scale=cfg_scale,
                suppress_tokens=tokenizer.convert_tokens_to_ids(model.config.mm_special_tokens),
            )
            
            # Tokenize input
            tokens = tokenizer(
                [prompt + image_prefix],
                return_tensors='pt',
                padding='longest',
                padding_side='left',
            )
            input_ids = tokens.input_ids.cuda()
            attention_mask = tokens.attention_mask.cuda()
            negative_ids = tokenizer.encode(
                image_prefix,
                add_special_tokens=False,
                return_tensors='pt',
            ).cuda().expand(1, -1)
            
            # Generate
            print("Starting token generation...")
            torch.manual_seed(seed)
            tokens = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                negative_prompt_ids=negative_ids,
            )
            
            progress(0.8, desc="Decoding image...")
            print("Decoding tokens to image...")
            
            tokens = torch.nn.functional.pad(tokens, (0, 1), value=tokenizer.convert_tokens_to_ids('<EOM>'))
            torch.manual_seed(seed)
            _, images = model.mmdecode(tokenizer, tokens[0], skip_special_tokens=False)
            
            if images and len(images) > 0:
                print(f"Successfully generated image: {type(images[0])}")
                progress(1.0, desc="Complete!")
                return images[0]
            else:
                print("No images generated")
                return None
            
        except Exception as e:
            import traceback
            error_msg = f"Error in generate_image_en: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None
    
    def generate_image_zh(self, prompt, image_size, cfg_scale, min_p, seed, progress=gr.Progress()):
        """Generate image using Chinese model"""
        try:
            print(f"Starting Chinese image generation with prompt: {prompt[:50]}...")
            progress(0.1, desc="Loading Chinese model...")
            model, tokenizer = self.load_model(self.zh_model_path, "zh")
            model.set_generation_mode('image')
            
            progress(0.3, desc="Processing prompt...")
            
            # Parse image size
            if isinstance(image_size, str):
                size_parts = image_size.split('x')
                if len(size_parts) == 2:
                    width, height = int(size_parts[0]), int(size_parts[1])
                else:
                    width = height = int(image_size)
            else:
                width = height = image_size
            
            image_size_tuple = (width, height)
            downsample_size = 16
            
            token_h, token_w = image_size_tuple[0] // downsample_size, image_size_tuple[1] // downsample_size
            image_prefix = f'<SOM>{token_h} {token_w}<IMAGE>'
            
            print(f"Image size: {image_size_tuple}, tokens: {token_h}x{token_w}")
            progress(0.5, desc="Generating image...")
            
            generation_config = GenerationConfig(
                max_new_tokens=token_h * token_w,
                do_sample=True,
                temperature=1.0,
                min_p=min_p,
                top_p=1.0,
                guidance_scale=cfg_scale,
                suppress_tokens=tokenizer.convert_tokens_to_ids(model.config.mm_special_tokens),
            )
            
            # Tokenize input
            tokens = tokenizer(
                [prompt + image_prefix],
                return_tensors='pt',
                padding='longest',
                padding_side='left',
            )
            input_ids = tokens.input_ids.cuda()
            attention_mask = tokens.attention_mask.cuda()
            negative_ids = tokenizer.encode(
                image_prefix,
                add_special_tokens=False,
                return_tensors='pt',
            ).cuda().expand(1, -1)
            
            # Generate
            print("Starting token generation...")
            torch.manual_seed(seed)
            tokens = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                negative_prompt_ids=negative_ids,
            )
            
            progress(0.8, desc="Decoding image...")
            print("Decoding tokens to image...")
            
            tokens = torch.nn.functional.pad(tokens, (0, 1), value=tokenizer.convert_tokens_to_ids('<EOM>'))
            torch.manual_seed(seed)
            _, images = model.mmdecode(tokenizer, tokens[0], skip_special_tokens=False)
            
            if images and len(images) > 0:
                print(f"Successfully generated image: {type(images[0])}")
                progress(1.0, desc="Complete!")
                return images[0]
            else:
                print("No images generated")
                return None
            
        except Exception as e:
            import traceback
            error_msg = f"Error in generate_image_zh: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None
    
    def chat_with_image(self, image, prompt, progress=gr.Progress()):
        """Chat with image using multimodal capabilities (uses English model for stability)"""
        try:
            progress(0.1, desc="Loading model...")
            
            # Use English model for chat as it has proper chat_template
            model, tokenizer = self.load_model(self.en_model_path, "en")
            model.set_generation_mode('text')
            
            progress(0.3, desc="Processing image...")
            
            if image is None:
                return "Please upload an image first."
            
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert('RGB')
            else:
                image = image.convert('RGB')
            
            progress(0.5, desc="Analyzing image...")
            
            image_str = model.tokenize_image(image)
            message = [{'role': 'user', 'content': image_str + '\n' + prompt}]
            input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt')
            
            progress(0.7, desc="Generating response...")
            
            with torch.inference_mode():
                generation_config = GenerationConfig(
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=1024,
                    top_p=0.9,
                    top_k=50,
                    num_beams=1,
                    use_cache=True,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.encode('<|im_end|>')[0],
                    pad_token_id=0,
                )
                output_ids = model.generate(input_ids.cuda(), generation_config=generation_config)
                texts, _ = model.mmdecode(tokenizer, output_ids[:, input_ids.shape[1]: -1])
            
            progress(1.0, desc="Complete!")
            return texts[0]
            
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_image_en_with_status(self, prompt, image_size, cfg_scale, min_p, seed, progress=gr.Progress()):
        """Wrapper for English image generation with status"""
        try:
            result = self.generate_image_en(prompt, image_size, cfg_scale, min_p, seed, progress)
            if result is None:
                return None, "❌ Image generation failed. Check console for details."
            else:
                return result, "✅ Image generated successfully!"
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg

    def generate_image_zh_with_status(self, prompt, image_size, cfg_scale, min_p, seed, progress=gr.Progress()):
        """Wrapper for Chinese image generation with status"""
        try:
            result = self.generate_image_zh(prompt, image_size, cfg_scale, min_p, seed, progress)
            if result is None:
                return None, "❌ 图像生成失败。请检查控制台获取详细信息。"
            else:
                return result, "✅ 图像生成成功！"
        except Exception as e:
            import traceback
            error_msg = f"❌ 错误: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg

    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="X-Omni WebUI", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# X-Omni WebUI")
            gr.Markdown("A web interface for X-Omni multimodal AI model supporting image generation and multimodal chat.")
            
            with gr.Tabs():
                # English Image Generation Tab
                with gr.TabItem("🎨 English Image Generation"):
                    gr.Markdown("### Generate images from English text prompts")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            en_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="A formal letter document with a professional tone...",
                                lines=5,
                                value="A formal letter document with a professional tone. Create a document that includes a section starting with \"To, Mr. Edward Robertson,\" aligned to the left."
                            )
                            
                            with gr.Row():
                                en_image_size = gr.Dropdown(
                                    label="Image Size",
                                    choices=["512", "768", "1024", "1152", "512x768", "768x512", "1024x768", "768x1024"],
                                    value="1152"
                                )
                                en_cfg_scale = gr.Slider(
                                    label="CFG Scale",
                                    minimum=0.1,
                                    maximum=10.0,
                                    value=1.0,
                                    step=0.1
                                )
                            
                            with gr.Row():
                                en_min_p = gr.Slider(
                                    label="Min-p",
                                    minimum=0.01,
                                    maximum=0.1,
                                    value=0.03,
                                    step=0.01
                                )
                                en_seed = gr.Number(
                                    label="Seed",
                                    value=1234,
                                    precision=0
                                )
                            
                            en_generate_btn = gr.Button("Generate Image", variant="primary")
                        
                        with gr.Column(scale=1):
                            en_output = gr.Image(label="Generated Image")
                            en_status = gr.Textbox(
                                label="Status", 
                                lines=3, 
                                interactive=False,
                                value="Ready to generate images..."
                            )
                    
                    # English examples within the English tab
                    gr.Markdown("---")
                    gr.Markdown("### Example Prompts")
                    gr.Examples(
                        examples=[
                            ["A formal letter document with a professional tone. Create a document that includes a section starting with \"To, Mr. Edward Robertson,\" aligned to the left."],
                            ["A beautiful sunset over mountains with golden clouds"],
                            ["A modern minimalist office space with large windows"]
                        ],
                        inputs=[en_prompt]
                    )
                
                # Chinese Image Generation Tab
                with gr.TabItem("🎨 中文图像生成"):
                    gr.Markdown("### 使用中文提示词生成图像")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            zh_prompt = gr.Textbox(
                                label="提示词",
                                placeholder="生成一张雪中的紫禁城全景封面图...",
                                lines=5,
                                value="生成一张雪中的紫禁城全景封面图，作为北京冬季旅游指南的主题。画面以近景构图展现建筑，红墙金瓦被皑皑白雪覆盖。"
                            )
                            
                            with gr.Row():
                                zh_image_size = gr.Dropdown(
                                    label="图像尺寸",
                                    choices=["512", "768", "1024", "1152", "512x768", "768x512", "1024x768", "768x1024"],
                                    value="1152"
                                )
                                zh_cfg_scale = gr.Slider(
                                    label="CFG Scale",
                                    minimum=0.1,
                                    maximum=10.0,
                                    value=1.0,
                                    step=0.1
                                )
                            
                            with gr.Row():
                                zh_min_p = gr.Slider(
                                    label="Min-p",
                                    minimum=0.01,
                                    maximum=0.1,
                                    value=0.03,
                                    step=0.01
                                )
                                zh_seed = gr.Number(
                                    label="随机种子",
                                    value=1234,
                                    precision=0
                                )
                            
                            zh_generate_btn = gr.Button("生成图像", variant="primary")
                        
                        with gr.Column(scale=1):
                            zh_output = gr.Image(label="生成的图像")
                            zh_status = gr.Textbox(
                                label="状态", 
                                lines=3, 
                                interactive=False,
                                value="准备生成图像..."
                            )
                    
                    # Chinese examples within the Chinese tab
                    gr.Markdown("---")
                    gr.Markdown("### 示例提示词")
                    gr.Examples(
                        examples=[
                            ["生成一张雪中的紫禁城全景封面图，作为北京冬季旅游指南的主题。"],
                            ["一朵盛开的莲花在宁静的池塘中，清晨的阳光洒在水面上"],
                            ["现代化的中式茶室，简约优雅的设计风格"]
                        ],
                        inputs=[zh_prompt]
                    )
                
                # Multimodal Chat Tab
                with gr.TabItem("💬 Multimodal Chat"):
                    gr.Markdown("### Chat with images using X-Omni")
                    gr.Markdown("*Note: Uses English model for optimal chat template compatibility. Supports both English and Chinese questions.*")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            chat_image = gr.Image(
                                label="Upload Image",
                                type="pil"
                            )
                        
                        with gr.Column(scale=2):
                            chat_prompt = gr.Textbox(
                                label="Question",
                                placeholder="Describe the image in detail.",
                                lines=3,
                                value="Describe the image in detail."
                            )
                            chat_btn = gr.Button("Send", variant="primary")
                            chat_output = gr.Textbox(
                                label="Response",
                                lines=10,
                                interactive=False
                            )
                    
                    # Add example questions below the main interface
                    gr.Markdown("---")
                    gr.Markdown("### Example Questions")
                    gr.Examples(
                        examples=[
                            ["Describe the image in detail."],
                            ["请详细描述一下图中的内容"]
                        ],
                        inputs=[chat_prompt],
                        label="Click to use example questions"
                    )
            
            # Event handlers
            en_generate_btn.click(
                fn=self.generate_image_en_with_status,
                inputs=[en_prompt, en_image_size, en_cfg_scale, en_min_p, en_seed],
                outputs=[en_output, en_status]
            )
            
            zh_generate_btn.click(
                fn=self.generate_image_zh_with_status,
                inputs=[zh_prompt, zh_image_size, zh_cfg_scale, zh_min_p, zh_seed],
                outputs=[zh_output, zh_status]
            )
            
            chat_btn.click(
                fn=self.chat_with_image,
                inputs=[chat_image, chat_prompt],
                outputs=[chat_output]
            )
        
        return demo


def main():
    # Check if models exist
    required_paths = [
        "/workspace/XOmni/checkpoints/FLUX.1-dev",
        "/workspace/XOmni/checkpoints/X-Omni-En", 
        "/workspace/XOmni/checkpoints/X-Omni-Zh"
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Error: Model path {path} not found!")
            return
    
    # Initialize WebUI
    webui = XOmniWebUI()
    demo = webui.create_interface()
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()
