import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image

try:
    import gradio as gr
except Exception:
    raise RuntimeError("Gradio is required. Install with `pip install gradio`")

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKL, AutoTokenizer,
                               Qwen3ForCausalLM, ZImageControlTransformer2DModel)
from videox_fun.pipeline import ZImageControlPipeline
from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler

# Paths for checkpoints (user provided)
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
ZIMAGE_DIR = os.path.join(CHECKPOINTS_DIR, "Z-Image-Turbo")
CONTROL_UNION_DIR = os.path.join(CHECKPOINTS_DIR, "Z-Image-Turbo-Fun-Controlnet-Union")

# Default example control image (from predict_t2i_control.py)
DEFAULT_CONTROL_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset", "pose.jpg")

# Example images directory (from ControlNet model)
EXAMPLES_DIR = os.path.join(CONTROL_UNION_DIR, "asset")

# Example data: (control_image_path, prompt, negative_prompt)
EXAMPLES = [
    [
        os.path.join(EXAMPLES_DIR, "pose.jpg"),
        "一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。",
        "模糊, 变形, 文字",
    ],
    [
        os.path.join(EXAMPLES_DIR, "pose2.jpg"),
        "一位身穿黑色西装的商务男士站在现代化办公楼前，手持公文包，表情自信而专注。背景是玻璃幕墙反射的城市天际线。",
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
    [
        os.path.join(EXAMPLES_DIR, "hed.jpg"),
        "一位男子独自坐在昏暗的酒吧吧台前，面前摆放着酒杯和酒瓶。他低头沉思，灯光在玻璃杯上投下柔和的光影，整个场景弥漫着一种安静而略带忧郁的氛围。",
        "模糊, 变形, 低质量",
    ],
]

# Globals
pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessor models (lazy load)
pose_detector = None
depth_model = None

# Third-party model URLs
THIRD_PARTY_DIR = os.path.join(CHECKPOINTS_DIR, "Third_Party")
MODEL_URLS = {
    "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
    "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
    "ZoeD_M12_N.pt": "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt",
}


def download_model_if_needed(filename):
    """Download model file if not exists"""
    os.makedirs(THIRD_PARTY_DIR, exist_ok=True)
    filepath = os.path.join(THIRD_PARTY_DIR, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    if filename not in MODEL_URLS:
        raise ValueError(f"Unknown model file: {filename}")
    
    url = MODEL_URLS[filename]
    print(f"Downloading {filename} from {url}...")
    
    try:
        from torch.hub import download_url_to_file
        download_url_to_file(url, filepath, progress=True)
        print(f"Downloaded {filename} to {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        raise


# ==================== Preprocessor Functions ====================

def extract_canny(image, low_threshold=100, high_threshold=200):
    """Extract Canny edges from image"""
    if image is None:
        return None
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_colored)


def extract_hed(image):
    """Extract HED-like soft edges (using Canny with blur for similar effect)"""
    if image is None:
        return None
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Approximate HED with blurred Canny
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Make edges softer
    edges = cv2.dilate(edges, None)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_colored)


def extract_depth(image):
    """Extract depth map using ZoeDepth"""
    global depth_model
    
    if image is None:
        return None
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    try:
        from comfyui.annotator.zoe.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
        from comfyui.annotator.zoe.zoedepth.utils.config import get_config
        from einops import rearrange
        
        if depth_model is None:
            depth_model = ZoeDepth.build_from_config(get_config("zoedepth", "infer"))
            
            # Auto-download and load pretrained weights
            zoe_path = download_model_if_needed("ZoeD_M12_N.pt")
            depth_model.load_state_dict(torch.load(zoe_path, map_location="cpu")['model'], strict=False)
            
            depth_model = depth_model.to(device=device, dtype=torch.float32).eval()
        
        # Resize image
        h, w = image.shape[:2]
        scale = 512 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        with torch.no_grad():
            img_tensor = torch.from_numpy(resized).to(device, torch.float32) / 255.0
            img_tensor = rearrange(img_tensor, 'h w c -> 1 c h w')
            depth = depth_model.infer(img_tensor)
            depth = depth[0, 0].cpu().numpy()
            
            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)
            depth = (depth - vmin) / (vmax - vmin + 1e-8)
            depth = 1.0 - depth
            depth = (depth * 255).clip(0, 255).astype(np.uint8)
            
        # Resize back
        depth = cv2.resize(depth, (w, h))
        depth_colored = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_colored)
        
    except Exception as e:
        print(f"Depth extraction failed: {e}")
        print("Falling back to simple edge-based depth approximation")
        # Fallback: use gradient magnitude as pseudo-depth
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        depth_colored = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_colored)


def extract_pose(image):
    """Extract pose using DWPose"""
    global pose_detector
    
    if image is None:
        return None
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    try:
        from comfyui.annotator.dwpose_utils import DWposeDetector
        
        if pose_detector is None:
            # Auto-download ONNX models if needed
            onnx_det = download_model_if_needed("yolox_l.onnx")
            onnx_pose = download_model_if_needed("dw-ll_ucoco_384.onnx")
            
            pose_detector = DWposeDetector(onnx_det, onnx_pose)
        
        pose_image = pose_detector(image)
        return Image.fromarray(pose_image)
        
    except Exception as e:
        print(f"Pose extraction failed: {e}")
        return None


def run_preprocessor(image, preprocessor_type, canny_low, canny_high):
    """Run selected preprocessor on input image"""
    if image is None:
        return None
    
    if preprocessor_type == "Canny":
        return extract_canny(image, int(canny_low), int(canny_high))
    elif preprocessor_type == "HED":
        return extract_hed(image)
    elif preprocessor_type == "Depth":
        return extract_depth(image)
    elif preprocessor_type == "Pose":
        return extract_pose(image)
    else:
        return image


def find_controlnet_checkpoint():
    candidate = os.path.join(CONTROL_UNION_DIR, "Z-Image-Turbo-Fun-Controlnet-Union.safetensors")
    if os.path.exists(candidate):
        return candidate
    return None


def load_pipeline(model_dir=None, transformer_ckpt=None, weight_dtype=torch.bfloat16):
    global pipeline
    if pipeline is not None:
        return pipeline

    model_dir = model_dir or ZIMAGE_DIR
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config_path = os.path.join("config", "z_image", "z_image_control.yaml")
    if not os.path.exists(config_path):
        # fallback to packaged config in repo
        config_path = os.path.join(os.path.dirname(__file__), "config", "z_image", "z_image_control.yaml")

    config = OmegaConf.load(config_path)

    transformer = ZImageControlTransformer2DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)

    if transformer_ckpt is not None:
        if transformer_ckpt.endswith("safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(transformer_ckpt)
        else:
            state_dict = torch.load(transformer_ckpt, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        transformer.load_state_dict(state_dict, strict=False)

    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae").to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=weight_dtype, low_cpu_mem_usage=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")

    pipeline = ZImageControlPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )

    # Move pipeline to device (GPU if available)
    try:
        pipeline.to(device)
    except Exception:
        pipeline.to("cpu")

    return pipeline


def preprocess_control_image(image, size=(1728, 992)):
    """
    Preprocess control image to latent. If image is None, use default pose.jpg.
    The pipeline requires control_image to be provided (otherwise it errors).
    """
    # If no image provided, load default example
    if image is None:
        if os.path.exists(DEFAULT_CONTROL_IMAGE):
            image = Image.open(DEFAULT_CONTROL_IMAGE).convert("RGB")
        else:
            raise ValueError("Control image is required. Please upload an image or ensure asset/pose.jpg exists.")

    # Accept PIL image or numpy array
    if isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.fromarray(image)

    from videox_fun.utils.utils import get_image_latent

    # get_image_latent expects path or PIL; keep same size order used in repo
    latent = get_image_latent(pil, sample_size=[size[0], size[1]])
    # mimic original script slicing
    return latent[:, :, 0]


def generate(prompt: str,
             negative_prompt: str,
             guidance_scale: float,
             num_inference_steps: int,
             seed: int,
             height: int,
             width: int,
             control_image,
             control_context_scale: float,
             load_model: bool,
             progress=gr.Progress()):
    global pipeline
    start = time.time()

    transformer_ckpt = find_controlnet_checkpoint()
    if load_model or pipeline is None:
        progress(0, desc="Loading model...")
        pipeline = load_pipeline(model_dir=ZIMAGE_DIR, transformer_ckpt=transformer_ckpt)

    generator = torch.Generator(device=device if device != "cpu" else "cpu").manual_seed(seed)

    # Control image is required by the pipeline (otherwise it errors with undefined inpaint_latent)
    progress(0.1, desc="Preprocessing control image")
    control_latent = preprocess_control_image(control_image, size=(int(height), int(width)))

    progress(0.2, desc="Running inference")
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=guidance_scale,
            control_image=control_latent,
            num_inference_steps=num_inference_steps,
            control_context_scale=control_context_scale,
        ).images

    elapsed = time.time() - start
    # return first image
    out_img = result[0]
    return out_img


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# VideoX-Fun — Z-Image Text→Image (control)")
        
        with gr.Tabs():
            # ==================== Tab 1: Generate ====================
            with gr.TabItem("🎨 生成 Generate"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label="Prompt", value="一位年轻女子站在阳光明媚的海岸线上...", lines=4)
                        negative_prompt = gr.Textbox(label="Negative Prompt", value="模糊, 变形, 文字", lines=2)
                        # Load default example image if available
                        default_img = DEFAULT_CONTROL_IMAGE if os.path.exists(DEFAULT_CONTROL_IMAGE) else None
                        control_image = gr.Image(type="pil", label="Control Image (pose/condition)", value=default_img)
                        load_model = gr.Checkbox(label="Load model before generation (recommended)", value=True)
                        btn = gr.Button("Generate", variant="primary")

                    with gr.Column():
                        guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Guidance Scale")
                        steps = gr.Slider(1, 50, value=9, step=1, label="Inference Steps")
                        seed = gr.Number(value=42, label="Seed")
                        height = gr.Number(value=1728, label="Height (px)")
                        width = gr.Number(value=992, label="Width (px)")
                        control_scale = gr.Slider(0.0, 2.0, value=0.75, step=0.01, label="Control Context Scale")
                        out = gr.Image(label="Result")

                btn.click(fn=generate,
                          inputs=[prompt, negative_prompt, guidance, steps, seed, height, width, control_image, control_scale, load_model],
                          outputs=[out])

                # Examples section
                gr.Markdown("### 示例 Examples")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[control_image, prompt, negative_prompt],
                    label="点击下方示例快速填充 (Click to load example)",
                )

            # ==================== Tab 2: Preprocessor ====================
            with gr.TabItem("🔧 预处理 Preprocessor"):
                gr.Markdown("""
                ### 图像预处理器
                上传一张普通图片，选择预处理类型，生成控制图（Canny边缘/深度图/姿态骨架等），然后可以用于生成新图。
                """)
                
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(type="pil", label="输入图片 Input Image")
                        preprocessor_type = gr.Radio(
                            choices=["Canny", "HED", "Depth", "Pose"],
                            value="Canny",
                            label="预处理类型 Preprocessor Type"
                        )
                        with gr.Row():
                            canny_low = gr.Slider(0, 255, value=100, step=1, label="Canny Low Threshold")
                            canny_high = gr.Slider(0, 255, value=200, step=1, label="Canny High Threshold")
                        preprocess_btn = gr.Button("运行预处理 Run Preprocessor", variant="primary")
                    
                    with gr.Column():
                        preprocessed_output = gr.Image(type="pil", label="预处理结果 Preprocessed Output")
                        send_to_generate_btn = gr.Button("📤 发送到生成页面 Send to Generate Tab", variant="secondary")
                
                preprocess_btn.click(
                    fn=run_preprocessor,
                    inputs=[input_image, preprocessor_type, canny_low, canny_high],
                    outputs=[preprocessed_output]
                )
                
                # Send preprocessed image to generate tab
                send_to_generate_btn.click(
                    fn=lambda x: x,
                    inputs=[preprocessed_output],
                    outputs=[control_image]
                )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    print("Starting Gradio app — visit http://127.0.0.1:7860")
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
