import gradio as gr
from diffsynth.pipelines.qwen_image import (
    QwenImagePipeline, ModelConfig,
    QwenImageUnit_Image2LoRAEncode, QwenImageUnit_Image2LoRADecode
)
from safetensors.torch import save_file
import torch
from PIL import Image
import os
from datetime import datetime
import glob

from diffsynth.utils.lora import merge_lora
from diffsynth.core.loader.file import load_state_dict

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# VRAM 配置
vram_config_disk_offload = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# ========== 全局 Pipeline ==========
style_pipe = None      # Style 模式专用
cfb_pipe = None        # Coarse+Fine+Bias 模式专用
gen_pipe = None        # 图片生成专用


def load_style_model():
    """加载 Style 模式所需的模型"""
    global style_pipe
    if style_pipe is not None:
        return
    print("正在加载 Style 模型...")
    style_pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=os.path.join(CHECKPOINTS_DIR, "General-Image-Encoders/SigLIP2-G384/model.safetensors"), **vram_config_disk_offload),
            ModelConfig(path=os.path.join(CHECKPOINTS_DIR, "General-Image-Encoders/DINOv3-7B/model.safetensors"), **vram_config_disk_offload),
            ModelConfig(path=os.path.join(CHECKPOINTS_DIR, "Qwen-Image-i2L/Qwen-Image-i2L-Style.safetensors"), **vram_config_disk_offload),
        ],
        tokenizer_config=None,
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    )
    print("✅ Style 模型加载完成！")


def load_cfb_model():
    """加载 Coarse+Fine+Bias 模式所需的模型"""
    global cfb_pipe
    if cfb_pipe is not None:
        return
    print("正在加载 Coarse+Fine+Bias 模型...")
    text_encoder_files = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "Qwen-Image/text_encoder/model-*.safetensors")))
    cfb_pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=text_encoder_files, **vram_config_disk_offload),
            ModelConfig(path=os.path.join(CHECKPOINTS_DIR, "General-Image-Encoders/SigLIP2-G384/model.safetensors"), **vram_config_disk_offload),
            ModelConfig(path=os.path.join(CHECKPOINTS_DIR, "General-Image-Encoders/DINOv3-7B/model.safetensors"), **vram_config_disk_offload),
            ModelConfig(path=os.path.join(CHECKPOINTS_DIR, "Qwen-Image-i2L/Qwen-Image-i2L-Coarse.safetensors"), **vram_config_disk_offload),
            ModelConfig(path=os.path.join(CHECKPOINTS_DIR, "Qwen-Image-i2L/Qwen-Image-i2L-Fine.safetensors"), **vram_config_disk_offload),
        ],
        tokenizer_config=None,
        # Coarse+Fine 需要 processor（Qwen2VLProcessor）来处理图像编码
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    )
    print("✅ Coarse+Fine+Bias 模型加载完成！")


def load_generation_model():
    """加载用于生成图片的 Qwen-Image 模型"""
    global gen_pipe
    if gen_pipe is not None:
        return
    print("正在加载图片生成模型...")
    transformer_files = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "Qwen-Image/transformer/diffusion_pytorch_model*.safetensors")))
    text_encoder_files = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "Qwen-Image/text_encoder/model-*.safetensors")))
    vae_files = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "Qwen-Image/vae/diffusion_pytorch_model*.safetensors")))
    model_configs = []
    if transformer_files:
        model_configs.append(ModelConfig(path=transformer_files, **vram_config_disk_offload))
    if text_encoder_files:
        model_configs.append(ModelConfig(path=text_encoder_files, **vram_config_disk_offload))
    if vae_files:
        model_configs.append(ModelConfig(path=vae_files, **vram_config_disk_offload))
    tokenizer_path = os.path.join(CHECKPOINTS_DIR, "Qwen-Image/tokenizer")
    tokenizer_config = ModelConfig(path=tokenizer_path) if os.path.exists(tokenizer_path) else None
    gen_pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    )
    print("✅ 图片生成模型加载完成！")


# ========== 工具函数 ==========

def parse_gallery_images(images):
    """将 Gradio Gallery 的输入解析为 PIL Image 列表"""
    pil_images = []
    if images is None:
        return pil_images
    for img in images:
        if isinstance(img, tuple):
            img = img[0]
        if isinstance(img, str):
            pil_images.append(Image.open(img).convert("RGB"))
        elif isinstance(img, Image.Image):
            pil_images.append(img.convert("RGB"))
    return pil_images


def load_style_examples(style_num):
    """加载 Style 示例图片 (assets/style/*)"""
    style_dir = os.path.join(CHECKPOINTS_DIR, f"Qwen-Image-i2L/assets/style/{style_num}")
    if not os.path.exists(style_dir):
        return None, f"❌ 目录不存在: {style_dir}"
    images = sorted([os.path.join(style_dir, f) for f in os.listdir(style_dir) if f.endswith('.jpg') and not f.startswith('image_')])
    return images, f"✅ 已加载 {len(images)} 张示例图片"


def load_lora_examples(lora_num):
    """加载 Coarse+Fine+Bias 示例图片 (assets/lora/*)"""
    lora_dir = os.path.join(CHECKPOINTS_DIR, f"Qwen-Image-i2L/assets/lora/{lora_num}")
    if not os.path.exists(lora_dir):
        return None, f"❌ 目录不存在: {lora_dir}"
    images = sorted([os.path.join(lora_dir, f) for f in os.listdir(lora_dir) if f.endswith('.jpg') and not f.startswith('image_')])
    return images, f"✅ 已加载 {len(images)} 张示例图片"


# ========== LoRA 生成 ==========

def generate_style_lora(images):
    """Style 模式：从图片生成 LoRA"""
    global style_pipe
    pil_images = parse_gallery_images(images)
    if not pil_images:
        return None, "❌ 请上传至少一张图片！"
    try:
        if style_pipe is None:
            load_style_model()
        with torch.no_grad():
            embs = QwenImageUnit_Image2LoRAEncode().process(style_pipe, image2lora_images=pil_images)
            lora = QwenImageUnit_Image2LoRADecode().process(style_pipe, **embs)["lora"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"lora_style_{timestamp}.safetensors")
        save_file(lora, output_path)
        return output_path, f"✅ Style LoRA 生成成功！\n路径: {output_path}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 生成失败: {str(e)}"


def generate_cfb_lora(images):
    """Coarse+Fine+Bias 模式：从图片生成 LoRA"""
    global cfb_pipe
    pil_images = parse_gallery_images(images)
    if not pil_images:
        return None, "❌ 请上传至少一张图片！"
    try:
        if cfb_pipe is None:
            load_cfb_model()
        with torch.no_grad():
            embs = QwenImageUnit_Image2LoRAEncode().process(cfb_pipe, image2lora_images=pil_images)
            lora = QwenImageUnit_Image2LoRADecode().process(cfb_pipe, **embs)["lora"]
        # 合并 Bias
        bias_path = os.path.join(CHECKPOINTS_DIR, "Qwen-Image-i2L/Qwen-Image-i2L-Bias.safetensors")
        if os.path.exists(bias_path):
            lora_bias = load_state_dict(bias_path, torch_dtype=torch.bfloat16, device="cuda")
            lora = merge_lora([lora, lora_bias])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"lora_cfb_{timestamp}.safetensors")
        save_file(lora, output_path)
        return output_path, f"✅ Coarse+Fine+Bias LoRA 生成成功！\n路径: {output_path}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 生成失败: {str(e)}"


# ========== 图片生成 ==========

def generate_image_with_lora(prompt, lora_path_text, height, width, steps, seed):
    """使用 LoRA 生成图片"""
    global gen_pipe
    try:
        if gen_pipe is None:
            load_generation_model()
        # 解析 LoRA 路径
        lora_path = lora_path_text.strip() if lora_path_text else None
        if not lora_path or not os.path.exists(lora_path):
            return None, "❌ 请填写有效的 LoRA 文件路径"
        # 清除旧的 LoRA，再加载新的
        gen_pipe.clear_lora()
        gen_pipe.load_lora(gen_pipe.dit, lora_path)
        # 生成图片
        seed_val = int(seed) if seed is not None and str(seed).strip() != "" else None
        image = gen_pipe(prompt, seed=seed_val, height=int(height), width=int(width), num_inference_steps=int(steps))
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"generated_{timestamp}.png")
        if isinstance(image, list):
            image = image[0]
        if hasattr(image, "save"):
            image.save(out_path)
        else:
            Image.fromarray(image).save(out_path)
        return out_path, f"✅ 图片生成成功: {out_path}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 生成失败: {str(e)}"


# ========== Gradio UI ==========

def create_ui():
    with gr.Blocks(title="Qwen-Image-i2L", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 Qwen-Image-i2L: Image to LoRA
        
        将一组图片转换为 LoRA 模型，用于风格迁移或内容保留，然后用生成的 LoRA 生成新图片。
        """)
        
        with gr.Tabs():
            # ==================== Tab 1: Style ====================
            with gr.Tab("🎨 Style（风格迁移）"):
                gr.Markdown("""
                ### Style 模式
                - **特点**：提取图片风格（弱细节），适合风格迁移
                - **推荐**：3~5 张风格统一的图片
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 示例风格")
                        style_dropdown = gr.Dropdown(
                            choices=["1 - Rough Sketch", "2 - Abstract Vector", "3 - Black & White Sketch", "4 - Blue Flat"],
                            label="选择示例",
                            value="1 - Rough Sketch"
                        )
                        style_load_btn = gr.Button("加载示例图片")
                        style_example_status = gr.Textbox(label="状态", interactive=False)
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🖼️ 上传风格图片")
                        style_gallery = gr.Gallery(label="风格图片", columns=5, rows=2, height="auto", interactive=True, type="filepath")
                        with gr.Row():
                            style_clear_btn = gr.Button("清空")
                            style_gen_lora_btn = gr.Button("🚀 生成 Style LoRA", variant="primary")
                
                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📥 LoRA 输出")
                        style_lora_file = gr.File(label="下载 LoRA")
                        style_lora_status = gr.Textbox(label="生成状态", interactive=False, lines=2)
                    with gr.Column():
                        gr.Markdown("#### 🖼️ 用 LoRA 生成图片")
                        style_prompt = gr.Textbox(label="Prompt", value="a cat", lines=1)
                        style_lora_path = gr.Textbox(label="LoRA 文件路径", value="", lines=1)
                        with gr.Row():
                            style_height = gr.Number(label="高度", value=1024)
                            style_width = gr.Number(label="宽度", value=1024)
                        with gr.Row():
                            style_steps = gr.Number(label="步数", value=50)
                            style_seed = gr.Number(label="种子", value=0)
                        style_gen_img_btn = gr.Button("🖼️ 生成图片", variant="primary")
                        style_output_img = gr.Image(label="生成结果")
                        style_img_status = gr.Textbox(label="状态", interactive=False, lines=2)
                
                # 事件绑定
                style_load_btn.click(
                    fn=lambda x: load_style_examples(x.split(" - ")[0]),
                    inputs=[style_dropdown],
                    outputs=[style_gallery, style_example_status]
                )
                style_clear_btn.click(fn=lambda: (None, ""), outputs=[style_gallery, style_lora_status])
                style_gen_lora_btn.click(fn=generate_style_lora, inputs=[style_gallery], outputs=[style_lora_file, style_lora_status])
                style_gen_img_btn.click(
                    fn=generate_image_with_lora,
                    inputs=[style_prompt, style_lora_path, style_height, style_width, style_steps, style_seed],
                    outputs=[style_output_img, style_img_status]
                )
            
            # ==================== Tab 2: Coarse+Fine+Bias ====================
            with gr.Tab("🔧 Coarse+Fine+Bias（内容保留）"):
                gr.Markdown("""
                ### Coarse+Fine+Bias 模式
                - **特点**：保留图片内容和细节，生成的 LoRA 可作为训练初始化权重
                - **推荐**：5~10 张或更多相似内容的图片
                - **说明**：会自动合并 Bias LoRA 以校正风格
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 示例数据")
                        cfb_dropdown = gr.Dropdown(
                            choices=["1 - Puppy Backpack", "2 - Teddy Bear", "3 - Blueberries"],
                            label="选择示例",
                            value="3 - Blueberries"
                        )
                        cfb_load_btn = gr.Button("加载示例图片")
                        cfb_example_status = gr.Textbox(label="状态", interactive=False)
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🖼️ 上传图片")
                        cfb_gallery = gr.Gallery(label="输入图片", columns=5, rows=2, height="auto", interactive=True, type="filepath")
                        with gr.Row():
                            cfb_clear_btn = gr.Button("清空")
                            cfb_gen_lora_btn = gr.Button("🚀 生成 Coarse+Fine+Bias LoRA", variant="primary")
                
                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📥 LoRA 输出")
                        cfb_lora_file = gr.File(label="下载 LoRA")
                        cfb_lora_status = gr.Textbox(label="生成状态", interactive=False, lines=2)
                    with gr.Column():
                        gr.Markdown("#### 🖼️ 用 LoRA 生成图片")
                        cfb_prompt = gr.Textbox(label="Prompt", value="a bowl of blueberries", lines=1)
                        cfb_lora_path = gr.Textbox(label="LoRA 文件路径", value="", lines=1)
                        with gr.Row():
                            cfb_height = gr.Number(label="高度", value=1024)
                            cfb_width = gr.Number(label="宽度", value=1024)
                        with gr.Row():
                            cfb_steps = gr.Number(label="步数", value=50)
                            cfb_seed = gr.Number(label="种子", value=0)
                        cfb_gen_img_btn = gr.Button("🖼️ 生成图片", variant="primary")
                        cfb_output_img = gr.Image(label="生成结果")
                        cfb_img_status = gr.Textbox(label="状态", interactive=False, lines=2)
                
                # 事件绑定
                cfb_load_btn.click(
                    fn=lambda x: load_lora_examples(x.split(" - ")[0]),
                    inputs=[cfb_dropdown],
                    outputs=[cfb_gallery, cfb_example_status]
                )
                cfb_clear_btn.click(fn=lambda: (None, ""), outputs=[cfb_gallery, cfb_lora_status])
                cfb_gen_lora_btn.click(fn=generate_cfb_lora, inputs=[cfb_gallery], outputs=[cfb_lora_file, cfb_lora_status])
                cfb_gen_img_btn.click(
                    fn=generate_image_with_lora,
                    inputs=[cfb_prompt, cfb_lora_path, cfb_height, cfb_width, cfb_steps, cfb_seed],
                    outputs=[cfb_output_img, cfb_img_status]
                )
        
    return demo


if __name__ == "__main__":
    # 启动时预加载 Style 模型（轻量级，可快速启动）
    load_style_model()
    
    # 启动 Web 界面
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
