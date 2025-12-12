import os
import sys
import glob
import torch
import gradio as gr
import numpy as np
from PIL import Image
from functools import partial
import tempfile
import shutil

# Store project root and change to video-generation directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_GEN_DIR = os.path.join(PROJECT_ROOT, "video-generation")

# Add video-generation to Python path
sys.path.insert(0, VIDEO_GEN_DIR)

# Change working directory to video-generation (required for relative paths in the codebase)
os.chdir(VIDEO_GEN_DIR)
print(f"Working directory: {os.getcwd()}")

from infer_preprocess import process_one
from infer_utils import resizecrop
from inference_14b import build_pipe, read_pose_video, build_split_plan, resizecrop
import imageio
import decord

# Configuration (paths relative to video-generation directory)
MODEL_PATH = "../checkpoints/One-to-All-14b"
PRETRAINED_MODEL_PATH = "../pretrained_models/Wan2.1-T2V-14B-Diffusers"
VAE_PATH = "../pretrained_models/Wan2.1-T2V-14B-Diffusers/vae"
CONFIG_PATH = "./configs/wan2.1_t2v_14b.json"
CACHE_BASE_DIR = "../input_cache"
OUTPUT_BASE_DIR = "../output"
MAX_SHORT = 576

MAIN_CHUNK = 81
OVERLAP_FRAMES = 5
FINAL_CHUNK_CANDIDATES = [65, 69, 73, 77, 81]

negative_prompt = [
    "black background, Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
]

# Global pipeline
pipe = None

def initialize_pipeline():
    """Initialize the model pipeline"""
    global pipe
    if pipe is not None:
        return
    
    print("Initializing pipeline...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    from opensora.sample.pipeline_wanx_vhuman_tokenreplace import WanPipeline
    from opensora.model_variants.wanx_diffusers_src import WanTransformer3DModel_Refextractor_2D_Controlnet_prefix
    from opensora.encoder_variants import get_text_enc
    from opensora.vae_variants import get_vae
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file as safe_load
    
    model_dtype = torch.bfloat16
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=7.0,
        num_train_timesteps=1000,
        use_dynamic_shifting=False
    )
    
    vae = get_vae('wanx', VAE_PATH, model_dtype)
    encoders = get_text_enc('wanx-t2v', PRETRAINED_MODEL_PATH, model_dtype)
    text_encoder = encoders.text_encoder
    tokenizer = encoders.tokenizer
    
    model = WanTransformer3DModel_Refextractor_2D_Controlnet_prefix.from_config(CONFIG_PATH).to(model_dtype)
    model.set_up_controlnet("./configs/wan2.1_t2v_14b_controlnet_1.json", model_dtype)
    model.set_up_refextractor("./configs/wan2.1_t2v_14b_refextractor_2d_withmask2.json", model_dtype)
    model.eval()
    model.requires_grad_(False)
    
    # Load checkpoint
    checkpoint = {}
    shard_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".safetensors")]
    for shard_file in sorted(shard_files):
        sd = safe_load(os.path.join(MODEL_PATH, shard_file), device='cpu')
        checkpoint.update(sd)
    model.load_state_dict(checkpoint, strict=True)
    
    pipe = WanPipeline(
        transformer=model,
        vae=vae.vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler
    )
    pipe.to(device, dtype=model_dtype)
    print("Pipeline initialized!")


def preprocess_inputs(reference_image, driving_video, frame_interval, do_align, align_mode, 
                     face_fade, head_fade, without_face, progress=gr.Progress()):
    """Preprocess reference image and driving video"""
    progress(0, desc="Preprocessing inputs...")
    
    # Save uploaded files to temp location
    temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    reference_image.save(temp_ref.name)
    shutil.copy(driving_video, temp_vid.name)
    
    # Process
    progress(0.5, desc="Extracting poses...")
    process_one(
        reference_path=temp_ref.name,
        video_path=temp_vid.name,
        frame_interval=frame_interval,
        do_align=do_align,
        alignmode=align_mode,
        face_change=face_fade,
        head_change=head_fade,
        without_face=without_face
    )
    
    # Find cache directory
    refname = os.path.splitext(os.path.basename(temp_ref.name))[0]
    vidname = os.path.splitext(os.path.basename(temp_vid.name))[0]
    
    pattern = os.path.join(CACHE_BASE_DIR, f"ref_{refname}_driven_{vidname}_align_{align_mode}_*")
    matches = glob.glob(pattern)
    
    # Cleanup temp files
    os.unlink(temp_ref.name)
    os.unlink(temp_vid.name)
    
    if not matches:
        raise ValueError(f"Cache directory not found: {pattern}")
    
    progress(1.0, desc="Preprocessing complete!")
    return matches[0]


def generate_video(reference_image, driving_video, prompt, frame_interval, do_align, align_mode,
                  face_fade, head_fade, without_face, ref_cfg, pose_cfg, progress=gr.Progress()):
    """Main inference function"""
    
    # Initialize pipeline if needed
    initialize_pipeline()
    
    # Preprocess inputs
    progress(0, desc="Starting preprocessing...")
    cache_dir = preprocess_inputs(reference_image, driving_video, frame_interval, do_align, 
                                  align_mode, face_fade, head_fade, without_face, progress)
    
    progress(0.3, desc="Loading processed data...")
    
    # Load processed data
    image_input = Image.open(os.path.join(cache_dir, "image_input.png")).convert("RGB")
    new_h, new_w = image_input.height, image_input.width
    
    if min(new_h, new_w) > MAX_SHORT:
        if new_h < new_w:
            scale = MAX_SHORT / new_h
            new_h, new_w = MAX_SHORT, int(new_w * scale)
        else:
            scale = MAX_SHORT / new_w
            new_w, new_h = MAX_SHORT, int(new_h * scale)
    
    new_h, new_w = int(new_h // 16 * 16), int(new_w // 16 * 16)
    transform_fn = partial(resizecrop, th=new_h, tw=new_w)
    image_input = transform_fn(image_input)
    
    pose_tensor, pose_fps = read_pose_video(os.path.join(cache_dir, "pose.mp4"), transform_fn)
    pose_input_img = Image.open(os.path.join(cache_dir, "pose_input.png")).convert("RGB")
    pose_input_img = transform_fn(pose_input_img)
    mask_input = Image.open(os.path.join(cache_dir, "mask_input.png")).convert("L")
    mask_input = transform_fn(mask_input)
    
    mask_np = np.array(mask_input, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).unsqueeze(2)
    
    src_pose_tensor = torch.from_numpy(np.array(pose_input_img)).unsqueeze(0).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
    src_pose_tensor = src_pose_tensor.unsqueeze(2)
    
    # Split video into chunks
    split_plan = build_split_plan(pose_tensor.shape[2])
    all_generated_frames_np = {}
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    progress(0.4, desc="Generating video...")
    
    # Process each chunk
    for idx, (start, end) in enumerate(split_plan):
        progress(0.4 + 0.5 * (idx / len(split_plan)), desc=f"Processing chunk {idx+1}/{len(split_plan)}...")
        
        sub_video = pose_tensor[:, :, start:end]
        prev_frames = None
        
        if start > 0:
            needed_idx = range(start, start + OVERLAP_FRAMES)
            if all(k in all_generated_frames_np for k in needed_idx):
                prev_frames = [Image.fromarray(all_generated_frames_np[k]) for k in needed_idx]
        
        output_chunk = pipe(
            image=image_input,
            image_mask=mask_tensor,
            control_video=sub_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=new_h,
            width=new_w,
            num_frames=end - start,
            image_guidance_scale=ref_cfg,
            pose_guidance_scale=pose_cfg,
            num_inference_steps=30,
            generator=torch.Generator(device=device).manual_seed(42),
            black_image_cfg=True,
            black_pose_cfg=True,
            controlnet_conditioning_scale=1.0,
            return_tensor=True,
            case1=False,
            token_replace=(prev_frames is not None),
            prev_frames=prev_frames,
            image_pose=src_pose_tensor
        ).frames
        
        output_chunk = (output_chunk[0].detach().cpu() / 2 + 0.5).float().clamp(0, 1).permute(1, 2, 3, 0).numpy()
        output_chunk = (output_chunk * 255).astype("uint8")
        
        for j in range(end - start):
            gidx = start + j
            all_generated_frames_np[gidx] = output_chunk[j]
    
    progress(0.9, desc="Combining frames...")
    
    # Combine frames with original pose overlay
    alpha = 0.6
    frames_combined = []
    src_uint8 = ((pose_tensor / 2 + 0.5).clamp(0, 1) * 255)[0].byte().permute(1, 2, 3, 0).numpy()
    sorted_idx = sorted(all_generated_frames_np.keys())
    
    for t in sorted_idx:
        src = src_uint8[t].astype(np.float32)
        pred = all_generated_frames_np[t].astype(np.float32)
        blended = (alpha * src + (1 - alpha) * pred).round().astype(np.uint8)
        concat = np.concatenate([blended, pred.astype(np.uint8)], axis=1)
        frames_combined.append(concat)
    
    # Save output video
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    imageio.mimwrite(output_path, frames_combined, fps=pose_fps, quality=8)
    
    progress(1.0, desc="Complete!")
    return output_path


# Gradio Interface
def create_interface():
    with gr.Blocks(title="One-to-All Animation") as demo:
        gr.Markdown("""
        # 🎭 One-to-All Animation
        
        Transform any reference image with any motion pattern! Upload a reference image and a driving video to generate character animation.
        
        **Note**: First run will take some time to initialize the model.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                reference_image = gr.Image(type="pil", label="Reference Image")
                driving_video = gr.Video(label="Driving Video")
                prompt = gr.Textbox(label="Prompt (optional)", 
                                   placeholder="e.g., a person dancing joyfully on a street...",
                                   lines=2)
                
                with gr.Accordion("Advanced Options", open=False):
                    frame_interval = gr.Slider(1, 5, value=1, step=1, 
                                              label="Frame Interval (sample every N frames)")
                    do_align = gr.Checkbox(value=True, label="Do Alignment (Retargeting)")
                    align_mode = gr.Radio(["ref", "pose"], value="ref", 
                                         label="Alignment Mode (ref: align to reference, pose: use original)")
                    face_fade = gr.Checkbox(value=True, label="Fade Facial Landmarks")
                    head_fade = gr.Checkbox(value=False, label="Fade Head Landmarks")
                    without_face = gr.Checkbox(value=False, label="Skip Drawing Facial Landmarks")
                    
                    ref_cfg = gr.Slider(0.5, 3.0, value=2.0, step=0.1, 
                                       label="Image Guidance Scale")
                    pose_cfg = gr.Slider(0.0, 3.0, value=1.5, step=0.1, 
                                        label="Pose Guidance Scale")
                
                generate_btn = gr.Button("🎬 Generate Animation", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### Output")
                output_video = gr.Video(label="Generated Animation")
        
        # Examples
        gr.Markdown("### 📸 Examples")
        gr.Examples(
            examples=[
                ["../examples/img.png", "../examples/vid.mp4", "", 1, True, "ref", True, False, False, 2.0, 1.5],
                ["../examples/musk.jpg", "../examples/vid2.mp4", "", 1, True, "ref", True, False, False, 2.0, 1.5],
                ["../examples/maodie.png", "../examples/vid2.mp4", "", 1, True, "ref", True, True, False, 2.0, 1.5],
                ["../examples/joker2_resize.png", "../examples/douyinvid5_v2.mp4", 
                 "a clown in a vibrant red suit dancing joyfully on a street, black shoes, with skyscrapers and neon lights in an urban city background, clear hand", 
                 1, False, "pose", True, True, False, 2.0, 1.5],
            ],
            inputs=[reference_image, driving_video, prompt, frame_interval, do_align, align_mode, 
                   face_fade, head_fade, without_face, ref_cfg, pose_cfg],
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[reference_image, driving_video, prompt, frame_interval, do_align, align_mode,
                   face_fade, head_fade, without_face, ref_cfg, pose_cfg],
            outputs=output_video
        )
        
        gr.Markdown("""
        ### 💡 Tips
        - **Alignment Mode**: Use "ref" for retargeted pose (recommended), "pose" for direct pose transfer
        - **Facial/Head Fade**: Enable for better identity consistency
        - **Prompt**: Especially helpful for pose mode to guide generation
        - **Guidance Scales**: Higher values = stronger conditioning (try 2.0/1.5 or 1.5/0 for faster inference)
        """)
    
    return demo


if __name__ == "__main__":
    # Get absolute path to examples directory
    examples_dir = os.path.join(PROJECT_ROOT, "examples")
    
    demo = create_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, 
                allowed_paths=[examples_dir])
