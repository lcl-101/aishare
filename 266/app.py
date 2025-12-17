import os
import sys
import time
import torch
from PIL import Image

try:
    import gradio as gr
except Exception:
    raise RuntimeError("Gradio is required. Install with `pip install gradio`")

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path)]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKL, AutoTokenizer,
                               Qwen3ForCausalLM, ZImageControlTransformer2DModel)
from videox_fun.pipeline import ZImageControlPipeline
from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
from videox_fun.utils.utils import get_image_latent

# ==================== Configuration ====================
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
ZIMAGE_DIR = os.path.join(CHECKPOINTS_DIR, "Z-Image-Turbo")
CONTROLNET_2_0_PATH = os.path.join(CHECKPOINTS_DIR, "Z-Image-Turbo-Fun-Controlnet-Union-2.0", "Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors")
CONFIG_PATH = "config/z_image/z_image_control_2.0.yaml"

# Example assets
EXAMPLES_DIR = os.path.join(CHECKPOINTS_DIR, "Z-Image-Turbo-Fun-Controlnet-Union-2.0", "asset")
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset")

# T2I Examples: [control_image, prompt, negative_prompt]
T2I_EXAMPLES = [
    [
        os.path.join(EXAMPLES_DIR, "pose.jpg"),
        "一位年轻女子站在阳光明媚的海岸线上，画面为全身竖构图，身体微微侧向右侧，左手自然下垂，右臂弯曲扶在腰间，她的手指清晰可见，站姿放松而略带羞涩。她身穿轻盈的白色连衣裙，裙摆在海风中轻轻飘动，布料半透、质感柔软。",
        "模糊, 变形, 文字",
    ],
    [
        os.path.join(EXAMPLES_DIR, "pose2.jpg"),
        "一位身穿黑色西装的商务男士站在现代化办公楼前，手持公文包，表情自信而专注。背景是玻璃幕墙反射的城市天际线，阳光明媚。",
        "模糊, 变形, 文字, 扭曲",
    ],
    [
        os.path.join(EXAMPLES_DIR, "canny.jpg"),
        "一只威风凛凛的花豹正面特写，锐利的眼睛直视前方，脸上布满独特的斑点花纹。背景是模糊的丛林绿叶，阳光透过树叶洒下斑驳的光影。",
        "模糊, 变形, 低质量",
    ],
    [
        os.path.join(EXAMPLES_DIR, "depth.jpg"),
        "一间温馨的咖啡厅内景，木质桌椅排列整齐，窗边摆放着绿植。阳光透过落地窗洒进来，空气中仿佛弥漫着咖啡的香气。",
        "模糊, 变形, 杂乱",
    ],
]

# I2I Inpaint Examples: [control_image, inpaint_image, mask_image, prompt, negative_prompt]
I2I_EXAMPLES = [
    [
        os.path.join(EXAMPLES_DIR, "pose.jpg"),
        os.path.join(ASSET_DIR, "8.png"),
        os.path.join(ASSET_DIR, "mask.png"),
        "一位年轻女子站在阳光明媚的海岸线上，画面为全身竖构图，身穿白色连衣裙，紫色长发随风飘动。",
        "模糊, 变形, 文字",
    ],
]

# Globals
pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.bfloat16


def load_pipeline():
    """Load the Z-Image ControlNet 2.0 pipeline"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    print("Loading Z-Image Turbo ControlNet 2.0 pipeline...")
    
    config = OmegaConf.load(CONFIG_PATH)
    
    # Load transformer
    transformer = ZImageControlTransformer2DModel.from_pretrained(
        ZIMAGE_DIR,
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)
    
    # Load ControlNet weights
    if os.path.exists(CONTROLNET_2_0_PATH):
        print(f"Loading ControlNet from: {CONTROLNET_2_0_PATH}")
        from safetensors.torch import load_file
        state_dict = load_file(CONTROLNET_2_0_PATH)
        state_dict = state_dict.get("state_dict", state_dict)
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"Loaded with missing keys: {len(m)}, unexpected keys: {len(u)}")
    else:
        raise FileNotFoundError(f"ControlNet model not found at {CONTROLNET_2_0_PATH}")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(ZIMAGE_DIR, subfolder="vae").to(weight_dtype)
    
    # Load tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained(ZIMAGE_DIR, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        ZIMAGE_DIR, 
        subfolder="text_encoder", 
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True
    )
    
    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ZIMAGE_DIR, subfolder="scheduler")
    
    # Create pipeline
    pipeline = ZImageControlPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
    
    # Move to device with CPU offload
    pipeline.enable_model_cpu_offload(device=device)
    
    print("Pipeline loaded successfully!")
    return pipeline


def generate_t2i_control(
    prompt: str,
    negative_prompt: str,
    control_image,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    control_context_scale: float,
    progress=gr.Progress()
):
    """Text-to-Image with Control"""
    global pipeline
    
    try:
        if pipeline is None:
            progress(0, desc="Loading model...")
            pipeline = load_pipeline()
        
        if control_image is None:
            return None
        
        progress(0.2, desc="Preprocessing control image...")
        
        sample_size = [int(height), int(width)]
        
        # Preprocess control image
        control_latent = get_image_latent(control_image, sample_size=sample_size)[:, :, 0]
        
        # For T2I control, we still need to provide default inpaint_image and mask_image
        # (required by the pipeline even in pure T2I mode)
        inpaint_latent = torch.zeros([1, 3, sample_size[0], sample_size[1]])
        mask_latent = torch.ones([1, 1, sample_size[0], sample_size[1]]) * 255
        
        # Generate
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
        progress(0.3, desc="Generating image...")
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=int(height),
                width=int(width),
                generator=generator,
                guidance_scale=guidance_scale,
                image=inpaint_latent,
                mask_image=mask_latent,
                control_image=control_latent,
                num_inference_steps=int(num_inference_steps),
                control_context_scale=control_context_scale,
            ).images
        
        return result[0]
    
    except Exception as e:
        print(f"Error in T2I generation: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_i2i_inpaint(
    prompt: str,
    negative_prompt: str,
    control_image,
    inpaint_image,
    mask_image,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    control_context_scale: float,
    progress=gr.Progress()
):
    """Image-to-Image Inpainting with Control"""
    global pipeline
    
    try:
        if pipeline is None:
            progress(0, desc="Loading model...")
            pipeline = load_pipeline()
        
        progress(0.2, desc="Preprocessing images...")
        
        sample_size = [int(height), int(width)]
        
        # Preprocess images
        if inpaint_image is not None:
            inpaint_latent = get_image_latent(inpaint_image, sample_size=sample_size)[:, :, 0]
        else:
            inpaint_latent = torch.zeros([1, 3, sample_size[0], sample_size[1]])
        
        if mask_image is not None:
            mask_latent = get_image_latent(mask_image, sample_size=sample_size)[:, :1, 0]
        else:
            mask_latent = torch.ones([1, 1, sample_size[0], sample_size[1]]) * 255
        
        if control_image is not None:
            control_latent = get_image_latent(control_image, sample_size=sample_size)[:, :, 0]
        else:
            control_latent = None
        
        # Generate
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
        progress(0.3, desc="Generating inpainted image...")
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=int(height),
                width=int(width),
                generator=generator,
                guidance_scale=guidance_scale,
                image=inpaint_latent,
                mask_image=mask_latent,
                control_image=control_latent,
                num_inference_steps=int(num_inference_steps),
                control_context_scale=control_context_scale,
            ).images
        
        return result[0]
    
    except Exception as e:
        print(f"Error in I2I inpainting: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_from_imagemask(
    prompt: str,
    negative_prompt: str,
    edit_dict,
    control_image,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    control_context_scale: float,
    progress=gr.Progress()
):
    """Generate from ImageMask component
    
    ImageMask returns dict with:
    - 'background': the uploaded original image
    - 'layers': list of drawn layers (mask)
    - 'composite': combined image
    """
    global pipeline
    
    try:
        if pipeline is None:
            progress(0, desc="Loading model...")
            pipeline = load_pipeline()
        
        if edit_dict is None:
            print("No edit_dict provided")
            return None
        
        progress(0.2, desc="Processing mask...")
        
        from PIL import Image
        
        # Get background (original image)
        input_image = edit_dict.get('background')
        if input_image is None:
            print("No background image")
            return None
        
        # Get mask from layers[0]
        layers = edit_dict.get('layers', [])
        
        if layers and len(layers) > 0:
            # The drawn layer - convert to grayscale mask
            layer = layers[0]
            if isinstance(layer, Image.Image):
                # Replace transparent areas with black, keep drawn (white) areas
                if layer.mode == 'RGBA':
                    # Create black background
                    mask_image = Image.new('L', layer.size, 0)
                    # Get alpha channel - where user drew
                    alpha = layer.split()[3]
                    # Where alpha > 0, set to white (areas to inpaint)
                    mask_image.paste(255, mask=alpha)
                else:
                    mask_image = layer.convert('L')
            else:
                mask_image = Image.new("L", input_image.size, 0)
        else:
            # No mask drawn - create empty mask
            mask_image = Image.new("L", input_image.size, 0)
        
        sample_size = [int(height), int(width)]
        
        inpaint_latent = get_image_latent(input_image, sample_size=sample_size)[:, :, 0]
        mask_latent = get_image_latent(mask_image, sample_size=sample_size)[:, :1, 0]
        
        if control_image is not None:
            control_latent = get_image_latent(control_image, sample_size=sample_size)[:, :, 0]
        else:
            control_latent = None
        
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
        progress(0.3, desc="Generating...")
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=int(height),
                width=int(width),
                generator=generator,
                guidance_scale=guidance_scale,
                image=inpaint_latent,
                mask_image=mask_latent,
                control_image=control_latent,
                num_inference_steps=int(num_inference_steps),
                control_context_scale=control_context_scale,
            ).images
        
        return result[0]
    
    except Exception as e:
        print(f"Error in imagemask inpainting: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_ui():
    with gr.Blocks(title="Z-Image Turbo 2.0") as demo:
        gr.Markdown("# Z-Image Turbo ControlNet 2.0")
        gr.Markdown("Three features: Text-to-Image Control, Image Inpainting & Draw Mask Inpainting")
        
        with gr.Tabs():
            # ==================== Tab 1: T2I Control ====================
            with gr.TabItem("🎨 Text-to-Image (Control)"):
                gr.Markdown("### Generate images from text prompts with control image guidance")
                
                with gr.Row():
                    with gr.Column():
                        t2i_prompt = gr.Textbox(
                            label="Prompt",
                            lines=5,
                            value="一位年轻女子站在阳光明媚的海岸线上，画面为全身竖构图，身体微微侧向右侧，左手自然下垂，右臂弯曲扶在腰间，她的手指清晰可见，站姿放松而略带羞涩。她身穿轻盈的白色连衣裙，裙摆在海风中轻轻飘动，布料半透、质感柔软。"
                        )
                        t2i_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value="模糊, 变形, 文字"
                        )
                        t2i_control = gr.Image(type="pil", label="Control Image (pose/edge/depth)")
                        t2i_btn = gr.Button("Generate", variant="primary", size="lg")
                    
                    with gr.Column():
                        with gr.Row():
                            t2i_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Guidance Scale")
                            t2i_steps = gr.Slider(1, 50, value=25, step=1, label="Inference Steps")
                        with gr.Row():
                            t2i_seed = gr.Number(value=43, label="Seed", precision=0)
                            t2i_control_scale = gr.Slider(0.0, 2.0, value=0.75, step=0.01, label="Control Scale")
                        with gr.Row():
                            t2i_height = gr.Number(value=1728, label="Height", precision=0)
                            t2i_width = gr.Number(value=992, label="Width", precision=0)
                        t2i_output = gr.Image(label="Generated Image", type="pil", height=600)
                
                t2i_btn.click(
                    fn=generate_t2i_control,
                    inputs=[
                        t2i_prompt, t2i_negative, t2i_control,
                        t2i_guidance, t2i_steps, t2i_seed,
                        t2i_height, t2i_width, t2i_control_scale
                    ],
                    outputs=[t2i_output]
                )
                
                # T2I Examples
                gr.Markdown("### 示例 Examples")
                gr.Examples(
                    examples=T2I_EXAMPLES,
                    inputs=[t2i_control, t2i_prompt, t2i_negative],
                    label="点击加载示例 (Click to load)",
                )
            
            # ==================== Tab 2: I2I Inpaint ====================
            with gr.TabItem("🖌️ Image Inpainting"):
                gr.Markdown("### Inpaint and modify images with control guidance")
                
                with gr.Row():
                    with gr.Column():
                        i2i_prompt = gr.Textbox(
                            label="Prompt",
                            lines=5,
                            value="一位年轻女子站在阳光明媚的海岸线上，画面为全身竖构图"
                        )
                        i2i_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value="模糊, 变形, 文字"
                        )
                        
                        with gr.Row():
                            i2i_inpaint = gr.Image(type="pil", label="Input Image (原始图片)")
                            i2i_mask = gr.Image(type="pil", label="Mask Image (遮罩图 - 白色=修复区域)")
                        
                        gr.Markdown("""
                        **遮罩说明 (Mask Tips):**
                        - 白色区域 = 需要重新生成的部分
                        - 黑色区域 = 保持不变的部分
                        - 可以用任何图像编辑软件制作遮罩（如 Photoshop, GIMP, 画图等）
                        """)
                        
                        i2i_control = gr.Image(type="pil", label="Control Image (optional)")
                        i2i_btn = gr.Button("Generate Inpainting", variant="primary", size="lg")
                    
                    with gr.Column():
                        with gr.Row():
                            i2i_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Guidance Scale")
                            i2i_steps = gr.Slider(1, 50, value=25, step=1, label="Inference Steps")
                        with gr.Row():
                            i2i_seed = gr.Number(value=43, label="Seed", precision=0)
                            i2i_control_scale = gr.Slider(0.0, 2.0, value=0.75, step=0.01, label="Control Scale")
                        with gr.Row():
                            i2i_height = gr.Number(value=1728, label="Height", precision=0)
                            i2i_width = gr.Number(value=992, label="Width", precision=0)
                        i2i_output = gr.Image(label="Inpainted Image", type="pil", height=600)
                
                i2i_btn.click(
                    fn=generate_i2i_inpaint,
                    inputs=[
                        i2i_prompt, i2i_negative, i2i_control,
                        i2i_inpaint, i2i_mask,
                        i2i_guidance, i2i_steps, i2i_seed,
                        i2i_height, i2i_width, i2i_control_scale
                    ],
                    outputs=[i2i_output]
                )
                
                # I2I Examples
                gr.Markdown("### 示例 Examples")
                gr.Examples(
                    examples=I2I_EXAMPLES,
                    inputs=[i2i_control, i2i_inpaint, i2i_mask, i2i_prompt, i2i_negative],
                    label="点击加载示例 (Click to load)",
                )
            
            # ==================== Tab 3: Draw Mask Inpainting ====================
            with gr.TabItem("✏️ Draw Mask Inpainting"):
                gr.Markdown("### 在图片上直接绘制遮罩")
                
                with gr.Row():
                    draw_mask = gr.ImageMask(
                        height=600,
                        sources=['upload', 'clipboard'],
                        type="pil",
                        brush=gr.Brush(
                            colors=["#FFFFFF"],
                            color_mode="fixed",
                        ),
                        label="Edit Image"
                    )
                
                with gr.Row():
                    with gr.Column():
                        draw_prompt = gr.Textbox(
                            label="Prompt",
                            lines=2,
                            placeholder="输入提示词",
                            value=""
                        )
                        draw_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=1,
                            value="blurry ugly bad"
                        )
                    with gr.Column():
                        draw_control = gr.Image(type="pil", label="Control Image (可选)")
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        draw_steps = gr.Slider(1, 50, value=25, step=1, label="Steps")
                        draw_control_scale = gr.Slider(0.0, 2.0, value=0.75, step=0.01, label="Control Scale")
                    with gr.Row():
                        draw_seed = gr.Number(value=43, label="Seed", precision=0)
                        draw_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Guidance Scale")
                    with gr.Row():
                        draw_height = gr.Number(value=1728, label="Height", precision=0)
                        draw_width = gr.Number(value=992, label="Width", precision=0)
                
                draw_btn = gr.Button("Generate Inpainting", variant="primary")
                draw_output = gr.Image(label="Generated Image", type="pil", height=600)
                
                draw_btn.click(
                    fn=generate_from_imagemask,
                    inputs=[
                        draw_prompt, draw_negative, draw_mask, draw_control,
                        draw_guidance, draw_steps, draw_seed,
                        draw_height, draw_width, draw_control_scale
                    ],
                    outputs=[draw_output]
                )
        
        gr.Markdown("""
        ### 使用说明 Instructions
        
        **Text-to-Image (Control):**
        - 上传控制图（姿态/边缘/深度图等）
        - 输入文本描述
        - 点击 Generate 生成图像
        
        **Image Inpainting:**
        - 上传要修复的图像 (Input Image)
        - 上传遮罩图 (Mask) - 白色区域将被重新生成
        - 可选：上传控制图指导生成
        - 输入文本描述想要的效果
        
        **Draw Mask Inpainting:**
        - 上传图片到编辑器
        - 用画笔直接在图片上标记需要修复的区域
        - 白色涂抹 = 重新生成
        - 点击 Generate Inpainting
        
        **Tips:**
        - Control Scale 控制条件图的影响强度 (0-2)
        - Guidance Scale 通常设为 0 获得最佳效果
        - Steps 建议 20-30 之间
        """)
    
    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("Z-Image Turbo ControlNet 2.0 Web UI")
    print("=" * 50)
    print(f"Model directory: {ZIMAGE_DIR}")
    print(f"ControlNet path: {CONTROLNET_2_0_PATH}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Check if required files exist
    if not os.path.exists(ZIMAGE_DIR):
        print(f"ERROR: Model directory not found: {ZIMAGE_DIR}")
        print("Please download Z-Image-Turbo model to checkpoints/")
        exit(1)
    
    if not os.path.exists(CONTROLNET_2_0_PATH):
        print(f"ERROR: ControlNet 2.0 not found: {CONTROLNET_2_0_PATH}")
        print("Please download Z-Image-Turbo-Fun-Controlnet-Union-2.0 to checkpoints/")
        exit(1)
    
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config file not found: {CONFIG_PATH}")
        exit(1)
    
    print("All required files found. Starting Gradio app...")
    
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
