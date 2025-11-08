#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import gradio as gr
import importlib as imp
from pathlib import Path
import random
import tempfile
import shutil
from PIL import Image
import torch
from tqdm import tqdm
import io

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try to import required modules with error handling
try:
    # Completely bypass flash attention import issues
    import sys
    
    # Remove flash_attn if it's already imported
    modules_to_remove = [key for key in sys.modules.keys() if 'flash_attn' in key]
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
    
    # Set environment variables to disable flash attention
    import os
    os.environ["FLASH_ATTENTION_DISABLE"] = "1"
    os.environ["DISABLE_FLASH_ATTN"] = "1"
    os.environ["USE_FLASH_ATTENTION"] = "0"
    
    # Import torch and transformers with specific configs
    import torch
    # Disable flash attention at transformers level
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    from src.utils.model_utils import build_emu3p5
    from src.utils.generation_utils import generate, multimodal_decode
    from src.utils.painting_utils import ProtoWriter
    from src.utils.input_utils import build_image, smart_resize
    from src.utils.vis_proto import main as vis_proto_main
    from src.proto import emu_pb as story_pb
    MODEL_IMPORTS_AVAILABLE = True
    IMPORT_ERROR = None
    print("✅ Model modules imported successfully!")
except Exception as e:
    MODEL_IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠️ Warning: Failed to import model modules: {e}")
    print("The app will start in demo mode without actual inference capability.")


class Emu3p5GradioApp:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vq_model = None
        self.model_loaded = False
        self.demo_mode = not MODEL_IMPORTS_AVAILABLE
        
        # Default model paths
        self.default_model_path = "./weights/Emu3.5"
        self.default_image_model_path = "./weights/Emu3.5-Image" 
        self.default_vq_path = "./weights/Emu3.5-VisionTokenizer"
        self.default_tokenizer_path = "./src/tokenizer_emu3_ibq"
        
        # Task configurations
        self.task_configs = {
            "t2i": {
                "name": "文本生成图像",
                "description": "根据文本描述生成图像",
                "model_path": self.default_image_model_path,
                "use_image": False,
                "max_new_tokens": 5120,
                "classifier_free_guidance": 5.0,
                "image_area": 1048576,
                "examples": [
                    "A lively comic-style illustration depicting two humorous cartoon dogs interacting near a freshly dug backyard hole",
                    "A serene landscape with mountains reflected in a crystal clear lake",
                    "A futuristic city skyline at sunset with flying cars",
                    "A cute robot playing with butterflies in a flower garden"
                ]
            },
            "x2i": {
                "name": "图像到图像", 
                "description": "基于参考图像和文本提示生成图像",
                "model_path": self.default_model_path,
                "use_image": True,
                "max_new_tokens": 5120,
                "classifier_free_guidance": 3.0,
                "image_area": 1048576,
                "examples": [
                    {
                        "prompt": "As shown in the second figure: The ripe strawberry rests on a green leaf in the garden. Replace the chocolate truffle in first image with ripe strawberry from 2nd image",
                        "reference_images": ["./assets/ref_0.png", "./assets/ref_1.png"]
                    },
                    {
                        "prompt": "Change the color of the object in the image to blue",
                        "reference_images": ["./assets/ref_0.png"]
                    },
                    {
                        "prompt": "Add flowers around the main subject in the image",
                        "reference_images": ["./assets/ref_img.png"]
                    }
                ]
            },
            "howto": {
                "name": "视觉指导",
                "description": "生成分步骤的可视化指导说明",
                "model_path": self.default_model_path,
                "use_image": False,
                "max_new_tokens": 32768,
                "classifier_free_guidance": 3.0,
                "image_area": 518400,
                "examples": [
                    "How to cook Shrimp, Celery, and Pork Dumplings",
                    "How to make origami crane step by step",
                    "How to plant a tomato garden",
                    "How to tie a necktie properly"
                ]
            },
            "story": {
                "name": "视觉叙事",
                "description": "生成包含文本和图像的可视化故事",
                "model_path": self.default_model_path,
                "use_image": False,
                "max_new_tokens": 32768,
                "classifier_free_guidance": 3.0,
                "image_area": 518400,
                "examples": [
                    "Tell a story about a clay astronaut exploring Mars and discovering a new continent hidden beneath the red dust",
                    "Imagine a heartwarming tale about a little hedgehog who overcomes his fear of the dark with the help of glowing fireflies",
                    "Create a story about a magical forest where animals can talk and help lost travelers find their way home",
                    "Tell the adventure of a young pirate who discovers a treasure map leading to an underwater kingdom"
                ]
            }
        }

    def load_model(self, task_type="story"):
        """Load the model for the specified task type"""
        if self.demo_mode:
            return "⚠️ Running in demo mode - model inference not available due to import issues."
        
        if self.model_loaded:
            return "模型已加载！"
        
        try:
            config = self.task_configs[task_type]
            model_path = config["model_path"]
            
            # Check if model path exists
            if not os.path.exists(model_path):
                return f"❌ Model path not found: {model_path}"
            
            self.model, self.tokenizer, self.vq_model = build_emu3p5(
                model_path,
                self.default_tokenizer_path,
                self.default_vq_path,
                vq_type="ibq",
                model_device="auto",
                vq_device="cuda:0"
            )
            
            self.model_loaded = True
            return f"{task_type} 任务模型加载成功！"
        except Exception as e:
            return f"模型加载错误：{str(e)}"

    def create_config(self, task_type, prompt, reference_images=None, use_image=False):
        """Create a dynamic configuration for inference"""
        config = self.task_configs[task_type].copy()
        
        # Create a temporary config object
        class TempConfig:
            pass
        
        cfg = TempConfig()
        
        # Basic config
        cfg.model_path = config["model_path"]
        cfg.vq_path = self.default_vq_path
        cfg.tokenizer_path = self.default_tokenizer_path
        cfg.vq_type = "ibq"
        cfg.task_type = task_type
        cfg.use_image = use_image
        cfg.hf_device = "auto"
        cfg.vq_device = "cuda:0"
        cfg.streaming = False
        cfg.unconditional_type = "no_text"
        cfg.classifier_free_guidance = config["classifier_free_guidance"]
        cfg.max_new_tokens = config["max_new_tokens"]
        cfg.image_area = config["image_area"]
        cfg.seed = random.randint(1, 10000)
        
        # Build template and unconditional prompt
        cfg.unc_prompt, cfg.template = self.build_unc_and_template(task_type, use_image)
        
        # Sampling parameters
        cfg.sampling_params = dict(
            use_cache=True,
            text_top_k=1024 if task_type != "howto" else 200,
            text_top_p=0.9 if task_type != "howto" else 0.8,
            text_temperature=1.0 if task_type != "howto" else 0.7,
            image_top_k=5120 if task_type in ["t2i", "x2i"] else 10240,
            image_top_p=1.0,
            image_temperature=1.0,
            top_k=131072,
            top_p=1.0,
            temperature=1.0,
            num_beams_per_group=1,
            num_beam_groups=1,
            diversity_penalty=0.0,
            max_new_tokens=cfg.max_new_tokens,
            guidance_scale=1.0,
            use_differential_sampling=True,
        )
        
        cfg.sampling_params["do_sample"] = cfg.sampling_params["num_beam_groups"] <= 1
        cfg.sampling_params["num_beams"] = cfg.sampling_params["num_beams_per_group"] * cfg.sampling_params["num_beam_groups"]
        
        # Special tokens
        cfg.special_tokens = dict(
            BOS="<|extra_203|>",
            EOS="<|extra_204|>",
            PAD="<|endoftext|>",
            EOL="<|extra_200|>",
            EOF="<|extra_201|>",
            TMS="<|extra_202|>",
            IMG="<|image token|>",
            BOI="<|image start|>",
            EOI="<|image end|>",
            BSS="<|extra_100|>",
            ESS="<|extra_101|>",
            BOG="<|extra_60|>",
            EOG="<|extra_61|>",
            BOC="<|extra_50|>",
            EOC="<|extra_51|>",
        )
        
        # Set special token IDs
        cfg.special_token_ids = {}
        for k, v in cfg.special_tokens.items():
            cfg.special_token_ids[k] = self.tokenizer.encode(v)[0]
        
        # Prepare prompts
        if use_image and reference_images:
            cfg.prompts = [(f"000", {"prompt": prompt, "reference_image": reference_images})]
        else:
            cfg.prompts = [(f"000", prompt)]
            
        return cfg

    def build_unc_and_template(self, task, with_image):
        """Build unconditional prompt and template"""
        task_str = task.lower()
        
        if task_str == 'howto':
            extra_system_prompt = ' Please generate a response with interleaved text and images.'
        else:
            extra_system_prompt = ''

        if task_str == 'x2i':
            # Special handling for x2i task
            if with_image:
                unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
                tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: <|IMAGE|>{question} ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
            else:
                unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
                tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question} ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
        else:
            if with_image:
                unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
                tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
            else:
                unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
                tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question} ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
        
        return unc_p, tmpl

    def inference_single(self, cfg, name, question):
        """Run inference for a single prompt"""
        torch.cuda.empty_cache()
        
        proto_writer = ProtoWriter()
        
        reference_image = None
        if not isinstance(question, str):
            if isinstance(question["reference_image"], list):
                reference_image = []
                for img_path in question["reference_image"]:
                    reference_image.append(Image.open(img_path).convert("RGB"))
            else:
                reference_image = Image.open(question["reference_image"]).convert("RGB")
            question = question["prompt"]
        
        proto_writer.clear()
        proto_writer.extend([["question", question]])
        if reference_image is not None:
            if isinstance(reference_image, list):
                for idx, img in enumerate(reference_image):
                    proto_writer.extend([[f"reference_image", img]])
            else:
                proto_writer.extend([["reference_image", reference_image]])

        prompt = cfg.template.format(question=question)
        
        if reference_image is not None:
            if isinstance(reference_image, list):
                image_str = ""
                for img in reference_image:
                    image_str += build_image(img, cfg, self.tokenizer, self.vq_model)
            else:
                image_str = build_image(reference_image, cfg, self.tokenizer, self.vq_model)
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            unc_prompt = cfg.unc_prompt

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
            BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([BOS, input_ids], dim=1)

        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        force_same_image_size = True
        if isinstance(reference_image, list) and len(reference_image) > 1:
            force_same_image_size = False
        
        for result_tokens in generate(cfg, self.model, self.tokenizer, input_ids, unconditional_ids, None, force_same_image_size):
            try:
                result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
                mm_out = multimodal_decode(result, self.tokenizer, self.vq_model)
                proto_writer.extend(mm_out)
            except Exception as e:
                raise Exception(f"Failed to generate token sequence: {e}")

        # Save to temporary file
        temp_dir = tempfile.mkdtemp()
        proto_path = os.path.join(temp_dir, f"{name}.pb")
        proto_writer.save(proto_path)
        
        return proto_path, temp_dir

    def process_protobuf_results(self, proto_path):
        """Process protobuf results and return images and text"""
        with open(proto_path, 'rb') as f:
            story = story_pb.Story()
            story.ParseFromString(f.read())

        results = []
        
        # Add question
        if story.question:
            results.append(("text", story.question))
        
        # Add reference images
        for ref_img_data in story.reference_images:
            img = Image.open(io.BytesIO(ref_img_data.image.image_data))
            results.append(("image", img))
        
        # Add generated content
        for c in story.clips:
            for s in c.segments:
                if s.asr and s.asr.strip():
                    results.append(("text", s.asr))
                
                for im in s.images:
                    img = Image.open(io.BytesIO(im.image.image_data))
                    results.append(("image", img))
        
        return results

    def run_inference(self, task_type, prompt, reference_images=None, progress=gr.Progress()):
        """Main inference function for Gradio"""
        if self.demo_mode:
            progress(0.5, desc="Demo mode - generating mock results...")
            # Return mock results for demo
            mock_text = [
                f"📝 Input prompt: {prompt}",
                "🎭 This is a demo response.",
                "⚠️ To see actual results, please fix the model import issues and restart the app."
            ]
            mock_images = []  # No images in demo mode
            progress(1.0, desc="Demo complete!")
            return "🎭 Demo mode - showing mock results", mock_text, mock_images
        
        if not self.model_loaded:
            load_msg = self.load_model(task_type)
            if "错误" in load_msg or "Error" in load_msg:
                return load_msg, [], []

        try:
            progress(0.1, desc="准备配置...")
            
            # Handle reference images
            use_image = reference_images is not None and len(reference_images) > 0
            ref_img_paths = []
            
            if use_image:
                # Handle uploaded image file paths
                ref_img_paths = reference_images if reference_images else []
            
            progress(0.2, desc="创建配置...")
            cfg = self.create_config(task_type, prompt, ref_img_paths if ref_img_paths else None, use_image)
            
            progress(0.3, desc="运行推理...")
            proto_path, temp_dir = self.inference_single(cfg, "result", cfg.prompts[0][1])
            
            progress(0.8, desc="处理结果...")
            results = self.process_protobuf_results(proto_path)
            
            # Separate text and images
            output_text = []
            output_images = []
            
            for result_type, content in results:
                if result_type == "text":
                    output_text.append(content)
                elif result_type == "image":
                    output_images.append(content)
            
            progress(1.0, desc="完成！")
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return "✅ 推理已成功完成！", output_text, output_images
            
        except Exception as e:
            return f"❌ 推理过程中出现错误：{str(e)}", [], []

    def create_gradio_interface(self):
        """Create the Gradio interface"""
        
        def create_task_interface(task_type):
            """Create interface for a specific task"""
            config = self.task_configs[task_type]
            
            with gr.Column():
                gr.Markdown(f"## {config['name']}")
                gr.Markdown(f"*{config['description']}*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="提示词",
                            placeholder=f"请在此输入您的{config['name']}提示词...",
                            lines=3
                        )
                        
                        # Reference images for tasks that support them
                        if config["use_image"]:
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("### 参考图像")
                                    reference_image_1 = gr.Image(
                                        label="参考图像 1",
                                        type="filepath",
                                        height=200,
                                        show_label=True
                                    )
                                    reference_image_2 = gr.Image(
                                        label="参考图像 2 (可选)",
                                        type="filepath", 
                                        height=200,
                                        show_label=True
                                    )
                            # Create a list to hold both images for processing
                            reference_images = [reference_image_1, reference_image_2]
                        else:
                            reference_images = None
                        
                        generate_btn = gr.Button(f"生成{config['name']}", variant="primary")
                        
                        # Example prompts
                        gr.Markdown("### 示例提示词：")
                        example_buttons = []
                        for i, example in enumerate(config["examples"]):
                            btn = gr.Button(f"示例 {i+1}", size="sm")
                            example_buttons.append((btn, example))
                    
                    with gr.Column(scale=2):
                        status_output = gr.Textbox(label="状态", interactive=False)
                        text_output = gr.JSON(label="生成的文本", visible=True)
                        image_output = gr.Gallery(
                            label="生成的图像",
                            columns=2,
                            rows=2,
                            height="auto"
                        )
                
                # Event handlers
                if config["use_image"]:
                    def on_generate(prompt, ref_img1, ref_img2):
                        # Collect non-None images
                        ref_imgs = []
                        if ref_img1 is not None:
                            ref_imgs.append(ref_img1)
                        if ref_img2 is not None:
                            ref_imgs.append(ref_img2)
                        return self.run_inference(task_type, prompt, ref_imgs if ref_imgs else None)
                    
                    generate_btn.click(
                        fn=on_generate,
                        inputs=[prompt_input, reference_image_1, reference_image_2],
                        outputs=[status_output, text_output, image_output]
                    )
                else:
                    def on_generate(prompt):
                        return self.run_inference(task_type, prompt, None)
                    
                    generate_btn.click(
                        fn=on_generate,
                        inputs=[prompt_input],
                        outputs=[status_output, text_output, image_output]
                    )
                
                # Example button handlers
                for btn, example_data in example_buttons:
                    if isinstance(example_data, dict) and "reference_images" in example_data:
                        # Handle examples with reference images (for x2i task)
                        def create_example_handler(example=example_data):
                            def on_example_click():
                                # Load reference images
                                img1 = None
                                img2 = None
                                if len(example["reference_images"]) > 0 and os.path.exists(example["reference_images"][0]):
                                    img1 = example["reference_images"][0]
                                if len(example["reference_images"]) > 1 and os.path.exists(example["reference_images"][1]):
                                    img2 = example["reference_images"][1]
                                return example["prompt"], img1, img2
                            return on_example_click
                        
                        btn.click(
                            fn=create_example_handler(),
                            outputs=[prompt_input, reference_image_1, reference_image_2] if config["use_image"] else [prompt_input]
                        )
                    else:
                        # Handle text-only examples
                        def create_text_example_handler(example=example_data):
                            def on_text_example_click():
                                return example
                            return on_text_example_click
                        
                        btn.click(
                            fn=create_text_example_handler(),
                            outputs=prompt_input
                        )
        
        # Create main interface with tabs
        with gr.Blocks(title="Emu3.5 多模态生成器", theme=gr.themes.Soft()) as demo:
            if self.demo_mode:
                gr.Markdown(
                    """
                    # 🎨 Emu3.5 多模态生成器 (演示模式)
                    
                    ⚠️ **当前运行在演示模式下**: 由于导入问题，模型推理功能已禁用。
                    
                    **常见问题和解决方案:**
                    - Flash-attention 兼容性: 尝试 `pip install flash-attn --no-build-isolation`  
                    - CUDA 版本不匹配: 确保 PyTorch 和 CUDA 版本匹配
                    - 依赖项缺失: 运行 `pip install -r requirements.txt`
                    
                    在问题解决之前，界面将显示模拟响应。
                    """
                )
            else:
                gr.Markdown(
                    """
                    # 🎨 Emu3.5 多模态生成器
                    
                    使用强大的 Emu3.5 模型生成图像、视觉故事和分步指导。
                    在下方选择任务类型并开始创作！
                    """
                )
            
            with gr.Tabs():
                # Text to Image tab
                with gr.Tab("🖼️ 文本生成图像"):
                    create_task_interface("t2i")
                
                # Any to Image tab  
                with gr.Tab("🔄 图像到图像"):
                    create_task_interface("x2i")
                
                # Visual Guidance tab
                with gr.Tab("🎯 视觉指导"):
                    create_task_interface("howto")
                
                # Visual Narrative tab
                with gr.Tab("📖 视觉叙事"):
                    create_task_interface("story")
            
            gr.Markdown(
                """
                ---
                
                ### 使用提示：
                - **文本生成图像**: 根据文本描述创建图像
                - **图像到图像**: 使用参考图像指导生成
                - **视觉指导**: 生成分步骤的视觉说明
                - **视觉叙事**: 创建图文并茂的故事
                
                ### 系统要求：
                - 确保模型权重已下载到 `./weights/` 目录
                - 确保 CUDA 可用以支持 GPU 加速
                """
            )
        
        return demo


def check_requirements():
    """Check if required packages are installed"""
    try:
        import gradio
        import torch
        import transformers
        print("✅ 所有必需的包都已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少必需的包: {e}")
        print("请使用以下命令安装依赖:")
        print("pip install -r requirements_gradio.txt")
        return False

def check_model_weights(weights_dir="./weights"):
    """Check if model weights are available"""
    weights_path = Path(weights_dir)
    
    required_models = [
        "Emu3.5",
        "Emu3.5-Image", 
        "Emu3.5-VisionTokenizer"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = weights_path / model
        if not model_path.exists():
            missing_models.append(model)
    
    if missing_models:
        print("⚠️  缺少模型权重:")
        for model in missing_models:
            print(f"   - {model}")
        print("\n请从 Hugging Face 下载模型:")
        print("https://huggingface.co/BAAI/Emu3.5")
        print("https://huggingface.co/BAAI/Emu3.5-Image")  
        print("https://huggingface.co/BAAI/Emu3.5-VisionTokenizer")
        return False
    else:
        print("✅ 所有模型权重都可用")
        return True

def main():
    """Main function to run the Gradio app"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Emu3.5 Gradio Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing (creates public URL)")
    parser.add_argument("--skip-checks", action="store_true", help="Skip requirement and model checks")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    print("🚀 启动 Emu3.5 Gradio 应用...")
    
    if not args.skip_checks:
        print("\n📋 检查环境依赖...")
        if not check_requirements():
            sys.exit(1)
        
        print("\n🎯 检查模型权重...")
        if not check_model_weights():
            print("\n您仍然可以启动应用，但没有模型时推理将会失败。")
            try:
                response = input("是否继续? (y/N): ")
                if response.lower() != 'y':
                    sys.exit(1)
            except KeyboardInterrupt:
                print("\n👋 用户取消")
                sys.exit(1)
    
    print(f"\n🌐 在 {args.host}:{args.port} 启动服务器")
    if args.share:
        print("🔗 已启用公共分享")
    
    # Create and launch the app
    try:
        app = Emu3p5GradioApp()
        demo = app.create_gradio_interface()
        
        # Launch the app
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()