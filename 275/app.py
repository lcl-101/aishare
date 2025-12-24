import os
import sys
import cv2
import numpy as np
import torch
import gradio as gr
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

# Setup project path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.insert(0, project_root) if project_root not in sys.path else None

from wan.models.face_align import FaceAlignment
from wan.models.face_model import FaceModel
from wan.models.pdf import FanEncoder, det_landmarks, get_drive_expression_pd_fgc
from wan.models.portrait_encoder import PortraitEncoder
from wan.pipeline.pipeline_wan_long import WanI2VLongPipeline
from wan.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
from wan.utils.fp8_optimization import convert_model_weight_to_float8, replace_parameters_by_name, convert_weight_dtype_wrapper
from wan.utils.lora_utils import merge_lora, unmerge_lora
from wan.utils.utils import filter_kwargs, simple_save_videos_grid
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# ========================== Configuration ==========================
# Model paths - using pre-downloaded checkpoints
WAN_MODEL_PATH = os.path.join(project_root, "checkpoints/Wan2.1-I2V-14B-720P")
FLASHPORTRAIT_PATH = os.path.join(project_root, "checkpoints/FlashPortrait")
CONFIG_PATH = os.path.join(project_root, "config/wan2.1/wan_civitai.yaml")

TRANSFORMER_PATH = os.path.join(FLASHPORTRAIT_PATH, "transformer.pt")
PORTRAIT_ENCODER_PATH = os.path.join(FLASHPORTRAIT_PATH, "portrait_encoder.pt")
DET_MODEL_PATH = os.path.join(FLASHPORTRAIT_PATH, "face_det.onnx")
ALIGNMENT_MODEL_PATH = os.path.join(FLASHPORTRAIT_PATH, "face_landmark.onnx")
PD_FPG_MODEL_PATH = os.path.join(FLASHPORTRAIT_PATH, "pd_fpg.pth")

# GPU memory mode options:
# - model_full_load: entire model on GPU
# - model_cpu_offload: model moved to CPU after use
# - sequential_cpu_offload: each layer moved to CPU after use (slower but saves more memory)
# - model_full_load_and_qfloat8: full load with float8 quantization
# - model_cpu_offload_and_qfloat8: cpu offload with float8 quantization
GPU_MEMORY_MODE = "model_cpu_offload"

# Default inference parameters
DEFAULT_MAX_SIZE = 720
DEFAULT_SUB_NUM_FRAMES = 201
DEFAULT_LATENTS_NUM_FRAMES = 51
DEFAULT_CONTEXT_OVERLAP = 30
DEFAULT_CONTEXT_SIZE = 51
DEFAULT_IP_SCALE = 1.0
DEFAULT_TEXT_CFG_SCALE = 1.0
DEFAULT_EMO_CFG_SCALE = 4.0
DEFAULT_FPS = 25
DEFAULT_SHIFT = 5
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_SEED = 42

# Use bfloat16 if supported, otherwise float16
WEIGHT_DTYPE = torch.bfloat16

# Output directory
OUTPUT_DIR = os.path.join(project_root, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Examples directory
EXAMPLES_DIR = os.path.join(project_root, "examples")

# ========================== Global Variables ==========================
pipeline = None
face_aligner = None
pd_fpg_motion = None
device = None


def find_replacement(a):
    """Find the nearest valid frame count."""
    while a > 0:
        if (a - 1) % 4 == 0:
            return a
        a -= 1
    return 0


def get_emo_feature(video_path, face_aligner, pd_fpg_motion, device=torch.device("cuda")):
    """Extract emotion features from driven video."""
    pd_fpg_motion = pd_fpg_motion.to(device)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_list = []
    ret, frame = cap.read()
    while ret:
        resized_frame = frame
        frame_list.append(resized_frame.copy())
        ret, frame = cap.read()
    cap.release()
    
    num_frames = len(frame_list)
    num_frames = find_replacement(num_frames)
    frame_list = frame_list[:num_frames]
    
    landmark_list = det_landmarks(face_aligner, frame_list)[1]
    emo_list = get_drive_expression_pd_fgc(pd_fpg_motion, frame_list, landmark_list, device)
    
    emo_feat_list = []
    head_emo_feat_list = []
    for emo in emo_list:
        headpose_emb = emo["headpose_emb"]
        eye_embed = emo["eye_embed"]
        emo_embed = emo["emo_embed"]
        mouth_feat = emo["mouth_feat"]
        emo_feat = torch.cat([eye_embed, emo_embed, mouth_feat], dim=1)
        head_emo_feat = torch.cat([headpose_emb, emo_feat], dim=1)
        emo_feat_list.append(emo_feat)
        head_emo_feat_list.append(head_emo_feat)
    
    emo_feat_all = torch.cat(emo_feat_list, dim=0)
    head_emo_feat_all = torch.cat(head_emo_feat_list, dim=0)
    return emo_feat_all, head_emo_feat_all, fps, num_frames


def load_models():
    """Load all required models."""
    global pipeline, face_aligner, pd_fpg_motion, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = OmegaConf.load(CONFIG_PATH)
    
    # Load transformer
    print("Loading transformer...")
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(WAN_MODEL_PATH, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
    )
    
    print(f"Loading portrait transformer from: {TRANSFORMER_PATH}")
    if TRANSFORMER_PATH.endswith("safetensors"):
        from safetensors.torch import load_file
        transformer_state_dict = load_file(TRANSFORMER_PATH)
    else:
        transformer_state_dict = torch.load(TRANSFORMER_PATH, map_location="cpu", weights_only=True)
    transformer_state_dict = transformer_state_dict.get("state_dict", transformer_state_dict)
    m, u = transformer.load_state_dict(transformer_state_dict, strict=False)
    print(f"Transformer missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(WAN_MODEL_PATH, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(WEIGHT_DTYPE)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(WAN_MODEL_PATH, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    
    # Load text encoder
    print("Loading text encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(WAN_MODEL_PATH, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
    )
    text_encoder = text_encoder.eval()
    
    # Load CLIP image encoder
    print("Loading CLIP image encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(WAN_MODEL_PATH, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(WEIGHT_DTYPE)
    clip_image_encoder = clip_image_encoder.eval()
    
    # Load face aligner
    print("Loading face aligner...")
    face_aligner = FaceModel(
        face_alignment_module=FaceAlignment(
            gpu_id=None,
            alignment_model_path=ALIGNMENT_MODEL_PATH,
            det_model_path=DET_MODEL_PATH,
        ),
        reset=False,
    )
    
    # Load PD FPG motion encoder
    print("Loading PD FPG motion encoder...")
    pd_fpg_motion = FanEncoder()
    pd_fpg_checkpoint = torch.load(PD_FPG_MODEL_PATH, map_location="cpu")
    pd_fpg_motion.load_state_dict(pd_fpg_checkpoint, strict=False)
    pd_fpg_motion = pd_fpg_motion.eval()
    
    # Load portrait encoder
    print("Loading portrait encoder...")
    portrait_encoder_state_dict = torch.load(PORTRAIT_ENCODER_PATH, map_location="cpu", weights_only=True)
    proj_prefix = "proj_model."
    mouth_prefix = "mouth_proj_model."
    emo_prefix = "emo_proj_model."
    portrait_encoder_state_dict_sub_proj_state = {}
    portrait_encoder_state_dict_sub_mouth_state = {}
    portrait_encoder_state_dict_sub_emo_state = {}
    
    for k, v in portrait_encoder_state_dict.items():
        if k.startswith(proj_prefix):
            new_key = k[len(proj_prefix):]
            portrait_encoder_state_dict_sub_proj_state[new_key] = v
        elif k.startswith(mouth_prefix):
            new_key = k[len(mouth_prefix):]
            portrait_encoder_state_dict_sub_mouth_state[new_key] = v
        elif k.startswith(emo_prefix):
            new_key = k[len(emo_prefix):]
            portrait_encoder_state_dict_sub_emo_state[new_key] = v
    
    portrait_encoder = PortraitEncoder(adapter_in_dim=768, adapter_proj_dim=2048)
    portrait_encoder.proj_model.load_state_dict(portrait_encoder_state_dict_sub_proj_state)
    portrait_encoder.mouth_proj_model.load_state_dict(portrait_encoder_state_dict_sub_mouth_state)
    portrait_encoder.emo_proj_model.load_state_dict(portrait_encoder_state_dict_sub_emo_state)
    portrait_encoder = portrait_encoder.eval()
    
    # Setup scheduler
    print("Setting up scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline = WanI2VLongPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
        portrait_encoder=portrait_encoder,
    )
    
    # Apply GPU memory optimization
    if GPU_MEMORY_MODE == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)
    
    print("All models loaded successfully!")
    return "Models loaded successfully!"


def generate_portrait_video(
    reference_image,
    driven_video,
    prompt,
    negative_prompt,
    seed,
    num_inference_steps,
    guidance_scale,
    emo_cfg_scale,
    max_size,
    progress=gr.Progress()
):
    """Generate portrait video from reference image and driven video."""
    global pipeline, face_aligner, pd_fpg_motion, device
    
    if pipeline is None:
        return None, "错误：模型未加载，请重启应用。"
    
    try:
        progress(0.1, desc="处理输入...")
        
        # Process reference image
        if isinstance(reference_image, str):
            image_start = Image.open(reference_image).convert("RGB")
        else:
            image_start = Image.fromarray(reference_image).convert("RGB")
        
        clip_image = image_start.copy()
        width, height = image_start.size
        scale = max_size / max(width, height)
        width, height = int(width * scale), int(height * scale)
        
        # Ensure dimensions are divisible by 16
        height_division_factor = 16
        width_division_factor = 16
        if height % height_division_factor != 0:
            height = (height + height_division_factor - 1) // height_division_factor * height_division_factor
        if width % width_division_factor != 0:
            width = (width + width_division_factor - 1) // width_division_factor * width_division_factor
        
        image_start = image_start.resize([width, height], Image.LANCZOS)
        clip_image = clip_image.resize([width, height], Image.LANCZOS)
        
        progress(0.2, desc="准备视频张量...")
        
        # Prepare input video tensor
        input_video = torch.tile(
            torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
            [1, 1, DEFAULT_SUB_NUM_FRAMES, 1, 1]
        ) / 255
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, 1:, ] = 255
        
        progress(0.3, desc="提取表情特征...")
        
        # Get driven video path
        if isinstance(driven_video, str):
            driven_video_path = driven_video
        else:
            driven_video_path = driven_video
        
        # Extract emotion features
        emo_feat_all, head_emo_feat_all, fps, num_frames = get_emo_feature(
            driven_video_path, face_aligner, pd_fpg_motion, device=device
        )
        emo_feat_all = emo_feat_all.unsqueeze(0)
        head_emo_feat_all = head_emo_feat_all.unsqueeze(0)
        
        progress(0.4, desc="生成视频中（可能需要较长时间）...")
        
        # Set random seed
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
        # Generate video
        with torch.no_grad():
            sample = pipeline(
                prompt,
                num_frames=num_frames,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps),
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
                shift=DEFAULT_SHIFT,
                context_overlap=DEFAULT_CONTEXT_OVERLAP,
                context_size=DEFAULT_CONTEXT_SIZE,
                latents_num_frames=DEFAULT_LATENTS_NUM_FRAMES,
                ip_scale=DEFAULT_IP_SCALE,
                head_emo_feat_all=head_emo_feat_all.to(device),
                sub_num_frames=DEFAULT_SUB_NUM_FRAMES,
                text_cfg_scale=DEFAULT_TEXT_CFG_SCALE,
                emo_cfg_scale=emo_cfg_scale,
            ).videos
            sample = sample[:, :, 1:]
        
        progress(0.9, desc="保存视频...")
        
        # Save output video
        index = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4')]) + 1
        output_filename = f"{str(index).zfill(8)}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        simple_save_videos_grid(sample, output_path, fps=fps)
        
        progress(1.0, desc="完成!")
        
        return output_path, f"视频生成成功！保存至：{output_path}"
    
    except Exception as e:
        import traceback
        error_msg = f"生成过程中出错：{str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def get_examples():
    """Get list of example cases."""
    examples = []
    for i in range(1, 7):
        case_dir = os.path.join(EXAMPLES_DIR, f"case-{i}")
        if os.path.exists(case_dir):
            ref_image = os.path.join(case_dir, "reference.png")
            driven_video = os.path.join(case_dir, "driven_video.mp4")
            if os.path.exists(ref_image) and os.path.exists(driven_video):
                examples.append([
                    ref_image,
                    driven_video,
                    "The person is talking",  # default prompt
                    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    DEFAULT_SEED,
                    DEFAULT_NUM_INFERENCE_STEPS,
                    DEFAULT_GUIDANCE_SCALE,
                    DEFAULT_EMO_CFG_SCALE,
                    DEFAULT_MAX_SIZE,
                ])
    return examples


# ========================== Gradio Interface ==========================
def create_ui():
    """Create Gradio UI."""
    with gr.Blocks(title="FlashPortrait - 人像动画生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 FlashPortrait - 人像动画生成器
        
        通过参考图片和驱动视频生成人像动画视频。
        
        **使用步骤：**
        1. 上传一张参考人像图片
        2. 上传一个驱动视频（包含面部表情/动作）
        3. 根据需要调整参数
        4. 点击"生成视频"创建你的人像动画
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                with gr.Group():
                    gr.Markdown("### 📥 输入")
                    reference_image = gr.Image(
                        label="参考人像图片",
                        type="filepath",
                        sources=["upload", "clipboard"],
                    )
                    driven_video = gr.Video(
                        label="驱动视频",
                        sources=["upload"],
                    )
                
                # Parameters section
                with gr.Group():
                    gr.Markdown("### 🎛️ 参数设置")
                    prompt = gr.Textbox(
                        label="提示词 (英文)",
                        value="The person is talking",
                        placeholder="Describe what the person is doing...",
                    )
                    negative_prompt = gr.Textbox(
                        label="负面提示词",
                        value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        lines=3,
                    )
                    
                    with gr.Row():
                        seed = gr.Number(label="随机种子", value=DEFAULT_SEED, precision=0)
                        num_inference_steps = gr.Slider(
                            label="推理步数",
                            minimum=10,
                            maximum=50,
                            value=DEFAULT_NUM_INFERENCE_STEPS,
                            step=1,
                        )
                    
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="引导强度",
                            minimum=1.0,
                            maximum=10.0,
                            value=DEFAULT_GUIDANCE_SCALE,
                            step=0.5,
                        )
                        emo_cfg_scale = gr.Slider(
                            label="表情控制强度",
                            minimum=1.0,
                            maximum=8.0,
                            value=DEFAULT_EMO_CFG_SCALE,
                            step=0.5,
                        )
                    
                    max_size = gr.Slider(
                        label="最大分辨率",
                        minimum=480,
                        maximum=1080,
                        value=DEFAULT_MAX_SIZE,
                        step=16,
                    )
                
                # Generate button
                generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Output section
                with gr.Group():
                    gr.Markdown("### 📤 输出")
                    output_video = gr.Video(label="生成的视频")
                    output_message = gr.Textbox(label="状态信息", interactive=False)
        
        # Examples section
        gr.Markdown("### 📚 示例")
        gr.Markdown("点击下方任意示例进行加载：")
        
        examples = get_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[
                    reference_image,
                    driven_video,
                    prompt,
                    negative_prompt,
                    seed,
                    num_inference_steps,
                    guidance_scale,
                    emo_cfg_scale,
                    max_size,
                ],
                outputs=[output_video, output_message],
                fn=generate_portrait_video,
                cache_examples=False,
            )
        
        # Event handlers
        generate_btn.click(
            fn=generate_portrait_video,
            inputs=[
                reference_image,
                driven_video,
                prompt,
                negative_prompt,
                seed,
                num_inference_steps,
                guidance_scale,
                emo_cfg_scale,
                max_size,
            ],
            outputs=[output_video, output_message],
        )
    
    return demo


if __name__ == "__main__":
    # 启动时自动加载模型
    print("=" * 50)
    print("正在加载模型，请稍候...")
    print("=" * 50)
    load_models()
    print("=" * 50)
    print("模型加载完成，启动 Web 界面...")
    print("=" * 50)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
