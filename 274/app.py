import os
import math
import time
import random
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import PIL.Image
import torch
import librosa
import soundfile as sf

from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video import context_parallel as context_parallel_module
from longcat_video.context_parallel import context_parallel_util

from longcat_video.audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from audio_separator.separator import Separator


def init_single_gpu_context_parallel():
    """Initialize context parallel for single GPU mode without distributed training"""
    # Directly set the global variables in context_parallel_util for single GPU mode
    context_parallel_util.cp_size = 1
    context_parallel_util.cp_rank = 0
    context_parallel_util.dp_size = 1
    context_parallel_util.dp_rank = 0
    context_parallel_util.dp_group = None
    context_parallel_util.cp_group = None
    context_parallel_util.dp_ranks = [0]
    context_parallel_util.cp_ranks = [0]
    print("[Single GPU Mode] Context parallel initialized: cp_size=1, cp_rank=0")


# Global variables for model caching
pipe_single = None
pipe_multi = None
vocal_separator = None
audio_output_dir_temp = None
device = None
models_loaded = {"single": False, "multi": False}


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    uid = timestamp_part + random_part
    return uid


def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    if source_path is None:
        return None
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None
        
    default_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    cmd = f"mv '{default_vocal_path}' '{target_path}'"
    os.system(cmd)    
    return target_path


def audio_prepare_multi(left_temp_vocal_path, right_temp_vocal_path, generate_duration, 
                        left_raw_speech_path, right_raw_speech_path, sample_rate=16000, audio_type='para'):
    """Prepare multi-person audio embeddings"""
    left_speech_array, right_speech_array = None, None
    left_raw_speech_array, right_raw_speech_array = None, None
    
    if left_temp_vocal_path is not None:
        left_speech_array, sr = librosa.load(left_temp_vocal_path, sr=sample_rate)
        left_raw_speech_array, _ = librosa.load(left_raw_speech_path, sr=sample_rate)
    
    if right_temp_vocal_path is not None:
        right_speech_array, sr = librosa.load(right_temp_vocal_path, sr=sample_rate)
        right_raw_speech_array, _ = librosa.load(right_raw_speech_path, sr=sample_rate)
    
    if left_speech_array is None:
        left_speech_array = np.zeros_like(right_speech_array)
        left_raw_speech_array = np.zeros_like(right_raw_speech_array)
    
    if right_speech_array is None:
        right_speech_array = np.zeros_like(left_speech_array)
        right_raw_speech_array = np.zeros_like(left_raw_speech_array)
    
    if audio_type == 'add':
        # Concatenation mode: person1 speaks first, then person2
        left_speech_array_ext = np.concatenate([left_speech_array, np.zeros_like(right_speech_array)])
        right_speech_array_ext = np.concatenate([np.zeros_like(left_speech_array), right_speech_array])
        merge_raw_speech = np.concatenate([left_raw_speech_array, np.zeros_like(right_raw_speech_array)]) + \
                           np.concatenate([np.zeros_like(left_raw_speech_array), right_raw_speech_array])
    elif audio_type == 'para':
        # Parallel mode: both speak at the same time
        left_speech_array_ext = left_speech_array
        right_speech_array_ext = right_speech_array
        merge_raw_speech = left_raw_speech_array + right_raw_speech_array
    else:
        raise NotImplementedError(f"Unsupported audio_type: {audio_type}")
    
    assert len(left_speech_array_ext) == len(right_speech_array_ext), "The two speech lengths should be equal"
    
    source_duration = len(left_speech_array_ext) / sample_rate
    added_sample_nums = math.ceil((generate_duration - source_duration) * sample_rate)
    if added_sample_nums > 0:
        left_speech_array_ext = np.append(left_speech_array_ext, [0.] * added_sample_nums)
        right_speech_array_ext = np.append(right_speech_array_ext, [0.] * added_sample_nums)
    
    return left_speech_array_ext, right_speech_array_ext, merge_raw_speech, source_duration


def load_models(checkpoint_dir="./checkpoints/LongCat-Video-Avatar", model_type="single"):
    """Load all required models"""
    global pipe_single, pipe_multi, vocal_separator, audio_output_dir_temp, device, models_loaded
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = 0
    torch_dtype = torch.bfloat16
    
    # Initialize context parallel for single GPU mode (bypass distributed training)
    init_single_gpu_context_parallel()
    cp_split_hw = context_parallel_util.get_optimal_split(1)
    
    # Load base model components
    base_model_dir = os.path.join(checkpoint_dir, '..', 'LongCat-Video')
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, subfolder="tokenizer", torch_dtype=torch_dtype)
    text_encoder = UMT5EncoderModel.from_pretrained(base_model_dir, subfolder="text_encoder", torch_dtype=torch_dtype)
    vae = AutoencoderKLWan.from_pretrained(base_model_dir, subfolder="vae", torch_dtype=torch_dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_dir, subfolder="scheduler", torch_dtype=torch_dtype)
    
    # Load DiT model based on type
    if model_type == "single":
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="avatar_single", cp_split_hw=cp_split_hw, torch_dtype=torch_dtype)
    else:
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="avatar_multi", cp_split_hw=cp_split_hw, torch_dtype=torch_dtype)
    
    # Load audio models
    wav2vec_path = os.path.join(checkpoint_dir, 'chinese-wav2vec2-base')
    audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(local_rank)
    audio_encoder.feature_extractor._freeze_parameters()
    
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path, local_files_only=True)
    
    # Setup vocal separator (only once)
    if vocal_separator is None:
        vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
        audio_output_dir_temp = Path("./audio_temp_file")
        os.makedirs(audio_output_dir_temp, exist_ok=True)
        
        audio_separator_model_path = os.path.dirname(vocal_separator_path)
        audio_separator_model_name = os.path.basename(vocal_separator_path)
        
        vocal_separator = Separator(
            output_dir=audio_output_dir_temp / "vocals",
            output_single_stem="vocals",
            model_file_dir=audio_separator_model_path,
        )
        vocal_separator.load_model(audio_separator_model_name)
    
    # Initialize pipeline
    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder=audio_encoder,
        wav2vec_feature_extractor=wav2vec_feature_extractor
    )
    pipe.to(local_rank)
    
    if model_type == "single":
        pipe_single = pipe
        models_loaded["single"] = True
    else:
        pipe_multi = pipe
        models_loaded["multi"] = True
    
    return f"{model_type.capitalize()} model loaded successfully!"


def load_single_model(checkpoint_dir):
    return load_models(checkpoint_dir, "single")


def load_multi_model(checkpoint_dir):
    return load_models(checkpoint_dir, "multi")


def generate_avatar_video(
    prompt,
    audio_file,
    image_file,
    stage_1,
    resolution,
    auto_segments,
    num_segments,
    num_inference_steps,
    text_guidance_scale,
    audio_guidance_scale,
    seed,
    ref_img_index,
    mask_frame_range,
    progress=gr.Progress()
):
    """Generate avatar video from audio and optional image"""
    global pipe_single, vocal_separator, audio_output_dir_temp, device
    
    if pipe_single is None:
        return None, "Error: Please load models first!"
    
    if audio_file is None:
        return None, "Error: Please upload an audio file!"
    
    if stage_1 == "ai2v" and image_file is None:
        return None, "Error: AI2V mode requires an image input!"
    
    progress(0.1, desc="Processing audio...")
    
    # Default parameters
    save_fps = 16
    num_frames = 93
    num_cond_frames = 13
    audio_stride = 2
    local_rank = 0
    
    negative_prompt = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    # 加入与相机运动相关的负面提示，避免镜头推进/变焦/平移等
    negative_prompt += ", zoom, zoom-in, zoom-out, camera zoom, camera movement, moving camera, dolly, push in, pull back, pan, tilt, tracking shot, camera motion, handheld, shaky camera, dolly zoom, crane shot, camera dolly"
    
    # Resolution settings
    if resolution == '480p':
        height, width = 480, 832
    elif resolution == '720p':
        height, width = 768, 1280
    
    # Extract vocal
    temp_vocal_path = extract_vocal_from_speech(
        audio_file, 
        f"/tmp/temp_speech_{generate_random_uid()}_vocal.wav", 
        vocal_separator, 
        audio_output_dir_temp
    )
    
    if temp_vocal_path is None or not os.path.exists(temp_vocal_path):
        return None, "Error: No vocal detected in the audio file!"
    
    progress(0.2, desc="Processing audio embedding...")
    
    # Load audio and get duration
    speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
    source_duration = len(speech_array) / sr
    
    # Auto-calculate number of segments based on audio duration
    if auto_segments:
        # First segment duration: num_frames / save_fps
        # Each additional segment adds: (num_frames - num_cond_frames) / save_fps
        first_segment_duration = num_frames / save_fps  # ~5.8125 seconds
        additional_segment_duration = (num_frames - num_cond_frames) / save_fps  # 5 seconds
        
        if source_duration <= first_segment_duration:
            num_segments = 1
        else:
            num_segments = 1 + math.ceil((source_duration - first_segment_duration) / additional_segment_duration)
        
        print(f"[Auto Segments] Audio duration: {source_duration:.2f}s, calculated segments: {num_segments}")
    
    # Audio padding
    generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
    added_sample_nums = math.ceil((generate_duration - source_duration) * sr)
    if added_sample_nums > 0:
        speech_array = np.append(speech_array, [0.] * added_sample_nums)
    
    # Audio embedding
    full_audio_emb = pipe_single.get_audio_embedding(speech_array, fps=save_fps * audio_stride, device=local_rank, sample_rate=sr)
    if torch.isnan(full_audio_emb).any():
        return None, "Error: Broken audio embedding with nan values!"
    
    # Cleanup temp vocal file
    if os.path.exists(temp_vocal_path):
        os.remove(temp_vocal_path)
    
    # Prepare audio embedding for the first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)
    
    # Set random seed
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)
    
    progress(0.3, desc=f"Generating segment 1/{num_segments}...")
    
    if stage_1 == 'at2v':
        # Audio-to-Video (AT2V)
        output_tuple = pipe_single.generate_at2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            resize_mode='default',
            audio_emb=audio_emb
        )
    elif stage_1 == 'ai2v':
        # Audio+Image-to-Video (AI2V)
        image = load_image(image_file)
        output_tuple = pipe_single.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            output_type='both',
            resize_mode='default',
            generator=generator,
            audio_emb=audio_emb
        )
    
    output, latent = output_tuple
    output = output[0]
    video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    video = [PIL.Image.fromarray(img) for img in video]
    del output
    torch_gc()
    
    # Get actual video dimensions from the generated frames
    actual_width, actual_height = video[0].size
    print(f"[Video Generation] Actual video size: {actual_width}x{actual_height}")
    
    # Long video generation
    all_generated_frames = video
    ref_latent = latent[:, :, :1].clone()
    current_video = video
    
    for segment_idx in range(1, num_segments):
        progress_val = 0.3 + 0.6 * (segment_idx / num_segments)
        progress(progress_val, desc=f"Generating segment {segment_idx + 1}/{num_segments}...")
        
        # Prepare audio embedding for next clip
        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)
        
        output_tuple = pipe_single.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=actual_height,
            width=actual_width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=False,
            resize_mode='default',
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range
        )
        output, latent = output_tuple
        
        output = output[0]
        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output
        
        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video
        torch_gc()
    
    progress(0.95, desc="Saving video...")
    
    # Save video
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "avatar_video")
    output_tensor = torch.from_numpy(np.array(all_generated_frames))
    save_video_ffmpeg(output_tensor, output_path, audio_file, fps=save_fps, quality=5)
    
    output_video_path = output_path + ".mp4"
    
    progress(1.0, desc="Done!")
    
    return output_video_path, f"Video generated successfully! Duration: {len(all_generated_frames) / save_fps:.2f}s"


def load_example(example_name):
    """Load example data"""
    import json
    
    example_path = f"assets/avatar/{example_name}.json"
    if not os.path.exists(example_path):
        return None, None, ""
    
    with open(example_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompt = data.get('prompt', '')
    image_path = data.get('cond_image', None)
    audio_path = data.get('cond_audio', {}).get('person1', None)
    
    return audio_path, image_path, prompt


def load_multi_example(example_name):
    """Load multi-person example data"""
    import json
    
    example_path = f"assets/avatar/{example_name}.json"
    if not os.path.exists(example_path):
        return None, None, None, "", "", ""
    
    with open(example_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompt = data.get('prompt', '')
    image_path = data.get('cond_image', None)
    audio1_path = data.get('cond_audio', {}).get('person1', None)
    audio2_path = data.get('cond_audio', {}).get('person2', None)
    audio_type = data.get('audio_type', 'para')
    
    # Format bbox info
    bbox_info = ""
    if 'bbox' in data:
        bbox1 = data['bbox'].get('person1', None)
        bbox2 = data['bbox'].get('person2', None)
        if bbox1:
            bbox_info += f"Person1: {bbox1}\n"
        if bbox2:
            bbox_info += f"Person2: {bbox2}"
    
    return audio1_path, audio2_path, image_path, prompt, audio_type, bbox_info


def generate_multi_avatar_video(
    prompt,
    audio_file1,
    audio_file2,
    image_file,
    audio_type,
    resolution,
    auto_segments,
    num_segments,
    num_inference_steps,
    text_guidance_scale,
    audio_guidance_scale,
    seed,
    ref_img_index,
    mask_frame_range,
    bbox_person1,
    bbox_person2,
    progress=gr.Progress()
):
    """Generate multi-person avatar video from two audio inputs and image"""
    global pipe_multi, vocal_separator, audio_output_dir_temp, device
    
    if pipe_multi is None:
        return None, "Error: Please load Multi-Person model first!"
    
    if audio_file1 is None and audio_file2 is None:
        return None, "Error: At least one audio file is required!"
    
    if image_file is None:
        return None, "Error: Reference image is required for multi-person mode!"
    
    progress(0.05, desc="Processing audio...")
    
    # Default parameters
    save_fps = 16
    num_frames = 93
    num_cond_frames = 13
    audio_stride = 2
    local_rank = 0
    sr = 16000
    
    negative_prompt = "Close-up, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    # 加入与相机运动相关的负面提示，避免镜头推进/变焦/平移等
    negative_prompt += ", zoom, zoom-in, zoom-out, camera zoom, camera movement, moving camera, dolly, push in, pull back, pan, tilt, tracking shot, camera motion, handheld, shaky camera, dolly zoom, crane shot, camera dolly"
    
    # Extract vocals
    left_temp_vocal_path = None
    right_temp_vocal_path = None
    
    if audio_file1:
        left_temp_vocal_path = extract_vocal_from_speech(
            audio_file1, 
            f"/tmp/temp_speech_{generate_random_uid()}_left_vocal.wav", 
            vocal_separator, 
            audio_output_dir_temp
        )
    
    if audio_file2:
        right_temp_vocal_path = extract_vocal_from_speech(
            audio_file2, 
            f"/tmp/temp_speech_{generate_random_uid()}_right_vocal.wav", 
            vocal_separator, 
            audio_output_dir_temp
        )
    
    if left_temp_vocal_path is None and right_temp_vocal_path is None:
        return None, "Error: No vocal detected in any audio file!"
    
    progress(0.15, desc="Processing audio embeddings...")
    
    # Calculate initial generate duration for auto segments
    first_segment_duration = num_frames / save_fps
    additional_segment_duration = (num_frames - num_cond_frames) / save_fps
    
    # Prepare multi-person audio
    generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
    
    left_speech_array_ext, right_speech_array_ext, merge_speech, source_duration = audio_prepare_multi(
        left_temp_vocal_path, right_temp_vocal_path, generate_duration,
        audio_file1, audio_file2, sample_rate=sr, audio_type=audio_type
    )
    
    # Auto-calculate segments based on audio duration
    if auto_segments:
        if source_duration <= first_segment_duration:
            num_segments = 1
        else:
            num_segments = 1 + math.ceil((source_duration - first_segment_duration) / additional_segment_duration)
        
        print(f"[Auto Segments] Audio duration: {source_duration:.2f}s, calculated segments: {num_segments}")
        
        # Recalculate generate duration and re-prepare audio
        generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
        left_speech_array_ext, right_speech_array_ext, merge_speech, _ = audio_prepare_multi(
            left_temp_vocal_path, right_temp_vocal_path, generate_duration,
            audio_file1, audio_file2, sample_rate=sr, audio_type=audio_type
        )
    
    # Save merged audio for final video
    merge_speech_path = f"/tmp/temp_speech_{generate_random_uid()}_merge.wav"
    sf.write(merge_speech_path, merge_speech, sr)
    
    # Get audio embeddings
    left_full_audio_emb = pipe_multi.get_audio_embedding(left_speech_array_ext, fps=save_fps * audio_stride, device=local_rank, sample_rate=sr)
    right_full_audio_emb = pipe_multi.get_audio_embedding(right_speech_array_ext, fps=save_fps * audio_stride, device=local_rank, sample_rate=sr)
    
    if torch.isnan(left_full_audio_emb).any() or torch.isnan(right_full_audio_emb).any():
        return None, "Error: Broken audio embedding with nan values!"
    
    # Cleanup temp vocal files
    if left_temp_vocal_path and os.path.exists(left_temp_vocal_path):
        os.remove(left_temp_vocal_path)
    if right_temp_vocal_path and os.path.exists(right_temp_vocal_path):
        os.remove(right_temp_vocal_path)
    
    # Prepare audio embedding for the first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=left_full_audio_emb.shape[0] - 1)
    left_audio_emb = left_full_audio_emb[center_indices][None, ...].to(local_rank)
    right_audio_emb = right_full_audio_emb[center_indices][None, ...].to(local_rank)
    audio_embs = torch.cat([left_audio_emb, right_audio_emb])
    
    # Set random seed
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)
    
    progress(0.2, desc="Preparing image and masks...")
    
    # Load image and prepare masks
    image = load_image(image_file)
    src_width, src_height = image.size
    
    # Define human / background mask
    background_mask = torch.zeros([src_height, src_width])
    human_mask1 = torch.zeros([src_height, src_width])
    human_mask2 = torch.zeros([src_height, src_width])
    
    # Parse bbox
    left_person_bbox = None
    right_person_bbox = None
    
    if bbox_person1 and bbox_person1.strip():
        try:
            left_person_bbox = [int(x.strip()) for x in bbox_person1.split(',')]
        except:
            pass
    
    if bbox_person2 and bbox_person2.strip():
        try:
            right_person_bbox = [int(x.strip()) for x in bbox_person2.split(',')]
        except:
            pass
    
    if left_person_bbox is None and right_person_bbox is None:
        # Default: split image in half
        face_scale = 0.1
        left_y_min, left_y_max = int(src_height * face_scale), int(src_height * (1 - face_scale))
        right_y_min, right_y_max = left_y_min, left_y_max
        half_width = src_width // 2
        left_x_min, left_x_max = int(half_width * face_scale), int(half_width * (1 - face_scale))
        right_x_min, right_x_max = int(half_width * face_scale + half_width), int(half_width * (1 - face_scale) + half_width)
    elif left_person_bbox is not None and right_person_bbox is not None:
        left_y_min, left_x_min, left_y_max, left_x_max = left_person_bbox
        right_y_min, right_x_min, right_y_max, right_x_max = right_person_bbox
    else:
        return None, "Error: Both person bboxes must be provided or both must be empty!"
    
    human_mask1[left_y_min:left_y_max, left_x_min:left_x_max] = 1
    human_mask2[right_y_min:right_y_max, right_x_min:right_x_max] = 1
    background_mask += human_mask1
    background_mask += human_mask2
    background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
    ref_target_masks = torch.stack([human_mask1, human_mask2, background_mask], dim=0).to(local_rank)
    
    progress(0.3, desc=f"Generating segment 1/{num_segments}...")
    
    # Generate first segment
    output_tuple = pipe_multi.generate_ai2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        output_type='both',
        resize_mode='default',
        generator=generator,
        audio_emb=audio_embs,
        ref_target_masks=ref_target_masks
    )
    
    output, latent = output_tuple
    output = output[0]
    video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    video = [PIL.Image.fromarray(img) for img in video]
    del output
    torch_gc()
    
    # Get actual video dimensions
    actual_width, actual_height = video[0].size
    print(f"[Video Generation] Actual video size: {actual_width}x{actual_height}")
    
    # Long video generation
    all_generated_frames = video
    ref_latent = latent[:, :, :1].clone()
    current_video = video
    
    for segment_idx in range(1, num_segments):
        progress_val = 0.3 + 0.6 * (segment_idx / num_segments)
        progress(progress_val, desc=f"Generating segment {segment_idx + 1}/{num_segments}...")
        
        # Prepare audio embedding for next clip
        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=left_full_audio_emb.shape[0] - 1)
        left_audio_emb = left_full_audio_emb[center_indices][None, ...].to(local_rank)
        right_audio_emb = right_full_audio_emb[center_indices][None, ...].to(local_rank)
        audio_embs = torch.cat([left_audio_emb, right_audio_emb])
        
        output_tuple = pipe_multi.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=actual_height,
            width=actual_width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
            resize_mode='default',
            audio_emb=audio_embs,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            ref_target_masks=ref_target_masks
        )
        output, latent = output_tuple
        
        output = output[0]
        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output
        
        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video
        torch_gc()
    
    progress(0.95, desc="Saving video...")
    
    # Save video
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "multi_avatar_video")
    output_tensor = torch.from_numpy(np.array(all_generated_frames))
    save_video_ffmpeg(output_tensor, output_path, merge_speech_path, fps=save_fps, quality=5)
    
    # Cleanup
    if os.path.exists(merge_speech_path):
        os.remove(merge_speech_path)
    
    output_video_path = output_path + ".mp4"
    
    progress(1.0, desc="Done!")
    
    return output_video_path, f"Video generated successfully! Duration: {len(all_generated_frames) / save_fps:.2f}s"


# Create Gradio interface
def create_ui():
    with gr.Blocks(title="LongCat Avatar 视频生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 LongCat Avatar 视频生成器
        
        基于音频输入生成说话人视频，支持单人和多人模式。
        """)
        
        with gr.Tabs():
            # ==================== 单人模式 Tab ====================
            with gr.TabItem("👤 单人模式"):
                gr.Markdown("""
                ### 单人音频转视频
                - **AI2V (音频+图片转视频)**: 基于音频、图片和文本提示生成视频
                - **AT2V (音频转视频)**: 仅基于音频和文本提示生成视频
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 输入")
                        
                        # 示例选择
                        single_example_dropdown = gr.Dropdown(
                            choices=["single_example_1"],
                            label="加载示例",
                            info="选择一个示例加载"
                        )
                        single_load_example_btn = gr.Button("📂 加载示例")
                        
                        single_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="描述场景和人物...",
                            lines=4,
                            value="A western man stands on stage under dramatic lighting, holding a microphone close to their mouth. Wearing a vibrant red jacket with gold embroidery, the singer is speaking while smoke swirls around them, creating a dynamic and atmospheric scene."
                        )
                        
                        single_audio_input = gr.Audio(
                            label="音频输入",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        single_image_input = gr.Image(
                            label="参考图片 (AI2V 模式必需)",
                            type="filepath"
                        )
                        
                        gr.Markdown("### ⚙️ 生成设置")
                        
                        single_stage_1 = gr.Radio(
                            choices=["ai2v", "at2v"],
                            value="ai2v",
                            label="生成模式",
                            info="AI2V: 音频+图片转视频 | AT2V: 音频转视频"
                        )
                        
                        single_resolution = gr.Radio(
                            choices=["480p", "720p"],
                            value="480p",
                            label="分辨率"
                        )
                        
                        single_auto_segments = gr.Checkbox(
                            label="根据音频时长自动计算分段数",
                            value=True,
                            info="根据音频长度自动确定视频分段数量"
                        )
                        
                        single_num_segments = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=1,
                            step=1,
                            label="分段数量 (手动)",
                            info="仅在关闭自动计算时使用，每段约 5 秒"
                        )
                        
                        with gr.Accordion("高级设置", open=False):
                            single_num_inference_steps = gr.Slider(
                                minimum=4,
                                maximum=100,
                                value=50,
                                step=1,
                                label="推理步数"
                            )
                            
                            single_text_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5,
                                label="文本引导强度"
                            )
                            
                            single_audio_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5,
                                label="音频引导强度"
                            )
                            
                            single_seed = gr.Number(
                                value=42,
                                label="随机种子",
                                precision=0
                            )
                            
                            single_ref_img_index = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
                                label="参考图像索引"
                            )
                            
                            single_mask_frame_range = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=6,
                                step=1,
                                label="遮罩帧范围"
                            )
                        
                        single_generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎥 输出")
                        single_output_video = gr.Video(label="生成的视频")
                        single_output_status = gr.Textbox(label="生成状态", interactive=False)
                        
                        gr.Markdown("### 📚 示例文件")
                        gr.Markdown("""
                        **可用示例：**
                        - `single_example_1`: 舞台上灯光下的西方男子
                        
                        **示例文件位置：**
                        - 图片: `assets/avatar/single/man.png`
                        - 音频: `assets/avatar/single/man.mp3`
                        """)
            
            # ==================== 多人模式 Tab ====================
            with gr.TabItem("👥 多人模式"):
                gr.Markdown("""
                ### 多人音频转视频
                生成两个人各自说话的视频，每人使用独立的音频轨道。
                
                **音频模式：**
                - **para (并行)**: 两人同时说话（音频时长需相同）
                - **add (串联)**: 人物1先说，然后人物2说
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 输入")
                        
                        # 示例选择
                        multi_example_dropdown = gr.Dropdown(
                            choices=["multi_example_1", "multi_example_2"],
                            label="加载示例",
                            info="选择一个示例加载"
                        )
                        multi_load_example_btn = gr.Button("📂 加载示例")
                        
                        multi_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="描述两人场景...",
                            lines=4,
                            value="Static camera, In a professional recording studio, two people stand facing each other, both wearing large headphones. They are speaking clearly into a large condenser microphone suspended between them."
                        )
                        
                        with gr.Row():
                            multi_audio_input1 = gr.Audio(
                                label="人物1 音频 (左)",
                                type="filepath",
                                sources=["upload", "microphone"]
                            )
                            multi_audio_input2 = gr.Audio(
                                label="人物2 音频 (右)",
                                type="filepath",
                                sources=["upload", "microphone"]
                            )
                        
                        multi_image_input = gr.Image(
                            label="参考图片 (必需)",
                            type="filepath"
                        )
                        
                        gr.Markdown("### ⚙️ 生成设置")
                        
                        multi_audio_type = gr.Radio(
                            choices=["para", "add"],
                            value="para",
                            label="音频模式",
                            info="para: 同时说话 | add: 人物1先说，然后人物2"
                        )
                        
                        multi_resolution = gr.Radio(
                            choices=["480p", "720p"],
                            value="480p",
                            label="分辨率"
                        )
                        
                        multi_auto_segments = gr.Checkbox(
                            label="根据音频时长自动计算分段数",
                            value=True,
                            info="根据音频长度自动确定视频分段数量"
                        )
                        
                        multi_num_segments = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=1,
                            step=1,
                            label="分段数量 (手动)",
                            info="仅在关闭自动计算时使用"
                        )
                        
                        with gr.Accordion("人物边界框 (可选)", open=False):
                            gr.Markdown("""
                            **格式:** `y_min, x_min, y_max, x_max`
                            
                            留空则自动检测（将图片对半分）。
                            """)
                            multi_bbox_person1 = gr.Textbox(
                                label="人物1 边界框",
                                placeholder="例如: 100, 80, 800, 640",
                                info="人物1（左侧）的边界框"
                            )
                            multi_bbox_person2 = gr.Textbox(
                                label="人物2 边界框", 
                                placeholder="例如: 50, 720, 820, 1300",
                                info="人物2（右侧）的边界框"
                            )
                        
                        with gr.Accordion("高级设置", open=False):
                            multi_num_inference_steps = gr.Slider(
                                minimum=4,
                                maximum=100,
                                value=50,
                                step=1,
                                label="推理步数"
                            )
                            
                            multi_text_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5,
                                label="文本引导强度"
                            )
                            
                            multi_audio_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5,
                                label="音频引导强度"
                            )
                            
                            multi_seed = gr.Number(
                                value=42,
                                label="随机种子",
                                precision=0
                            )
                            
                            multi_ref_img_index = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
                                label="参考图像索引"
                            )
                            
                            multi_mask_frame_range = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=6,
                                step=1,
                                label="遮罩帧范围"
                            )
                        
                        multi_generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎥 输出")
                        multi_output_video = gr.Video(label="生成的视频")
                        multi_output_status = gr.Textbox(label="生成状态", interactive=False)
                        
                        gr.Markdown("### 📚 示例文件")
                        gr.Markdown("""
                        **可用示例：**
                        - `multi_example_1`: 录音棚中的两人 (并行模式)
                        - `multi_example_2`: 咖啡馆中的两人对话 (串联模式，带边界框)
                        
                        **示例文件位置：**
                        - 图片: `assets/avatar/multi/sing.png`, `assets/avatar/multi/introduce.png`
                        - 音频: `assets/avatar/multi/sing_man.WAV`, `assets/avatar/multi/sing_woman.WAV` 等
                        """)
        
        # ==================== 事件处理 ====================
        # 单人模式事件
        single_load_example_btn.click(
            fn=load_example,
            inputs=[single_example_dropdown],
            outputs=[single_audio_input, single_image_input, single_prompt]
        )
        
        single_generate_btn.click(
            fn=generate_avatar_video,
            inputs=[
                single_prompt,
                single_audio_input,
                single_image_input,
                single_stage_1,
                single_resolution,
                single_auto_segments,
                single_num_segments,
                single_num_inference_steps,
                single_text_guidance_scale,
                single_audio_guidance_scale,
                single_seed,
                single_ref_img_index,
                single_mask_frame_range
            ],
            outputs=[single_output_video, single_output_status]
        )
        
        # 多人模式事件
        multi_load_example_btn.click(
            fn=load_multi_example,
            inputs=[multi_example_dropdown],
            outputs=[multi_audio_input1, multi_audio_input2, multi_image_input, multi_prompt, multi_audio_type, multi_bbox_person1]
        )
        
        multi_generate_btn.click(
            fn=generate_multi_avatar_video,
            inputs=[
                multi_prompt,
                multi_audio_input1,
                multi_audio_input2,
                multi_image_input,
                multi_audio_type,
                multi_resolution,
                multi_auto_segments,
                multi_num_segments,
                multi_num_inference_steps,
                multi_text_guidance_scale,
                multi_audio_guidance_scale,
                multi_seed,
                multi_ref_img_index,
                multi_mask_frame_range,
                multi_bbox_person1,
                multi_bbox_person2
            ],
            outputs=[multi_output_video, multi_output_status]
        )
    
    return demo


def load_all_models(checkpoint_dir="./checkpoints/LongCat-Video-Avatar"):
    """启动时加载所有模型"""
    print("=" * 50)
    print("正在加载模型，请稍候...")
    print("=" * 50)
    
    print("\n[1/2] 正在加载单人模式模型...")
    load_models(checkpoint_dir, "single")
    print("✓ 单人模式模型加载完成")
    
    print("\n[2/2] 正在加载多人模式模型...")
    load_models(checkpoint_dir, "multi")
    print("✓ 多人模式模型加载完成")
    
    print("\n" + "=" * 50)
    print("所有模型加载完成！")
    print("=" * 50)


if __name__ == "__main__":
    # 启动时自动加载模型
    load_all_models()
    
    # 创建并启动界面
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
