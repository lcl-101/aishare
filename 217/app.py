import os
import argparse
import datetime
import PIL.Image
import numpy as np
import cv2
from typing import Optional, Tuple, List
import tempfile
import shutil

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
from diffusers.utils import load_image, load_video
import gradio as gr

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


class LongCatVideoWebUI:
    def __init__(self, checkpoint_dir: str, enable_compile: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.enable_compile = enable_compile
        self.pipe = None
        self.local_rank = 0
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models and pipeline"""
        print("Initializing LongCat-Video models...")
        
        # Set up device
        if torch.cuda.is_available():
            self.local_rank = 0
            torch.cuda.set_device(self.local_rank)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
        
        # Setup distributed environment if not already set
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
        
        # Initialize distributed process group if using CUDA
        if torch.cuda.is_available():
            try:
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend="nccl" if torch.cuda.is_available() else "gloo",
                        timeout=datetime.timedelta(seconds=1800)
                    )
                    print("Distributed process group initialized")
            except Exception as e:
                print(f"Warning: Could not initialize distributed process group: {e}")
                print("Continuing with single GPU setup...")
        
        # Initialize context parallel
        try:
            init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)
            cp_size = context_parallel_util.get_cp_size()
            cp_split_hw = context_parallel_util.get_optimal_split(cp_size)
            print(f"Context parallel initialized: cp_size={cp_size}, cp_split_hw={cp_split_hw}")
        except Exception as e:
            print(f"Warning: Could not initialize context parallel: {e}")
            cp_split_hw = None

        # Load models
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
        
        print("Loading text encoder...")
        text_encoder = UMT5EncoderModel.from_pretrained(self.checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        
        print("Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(self.checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
        
        print("Loading scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
        
        print("Loading DiT...")
        dit_kwargs = {
            "torch_dtype": torch.bfloat16
        }
        if cp_split_hw is not None:
            dit_kwargs["cp_split_hw"] = cp_split_hw
            
        dit = LongCatVideoTransformer3DModel.from_pretrained(
            self.checkpoint_dir, 
            subfolder="dit", 
            **dit_kwargs
        )

        if self.enable_compile:
            print("Compiling DiT...")
            dit = torch.compile(dit)

        # Create pipeline
        self.pipe = LongCatVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
        )
        
        if torch.cuda.is_available():
            self.pipe.to(self.local_rank)
        
        print("Models initialized successfully!")

    def torch_gc(self):
        """Garbage collection for GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def generate_text_to_video(
        self, 
        prompt: str, 
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 42,
        enable_distill: bool = True,
        enable_refine: bool = True,
        progress=gr.Progress()
    ) -> List[str]:
        """Text to Video generation"""
        if not prompt.strip():
            return []
        
        progress(0, "Initializing generation...")
        
        generator = torch.Generator(device=self.local_rank)
        generator.manual_seed(seed)
        
        output_files = []
        
        # Default negative prompt if empty
        if not negative_prompt.strip():
            negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        try:
            # Stage 1: Basic T2V generation
            progress(0.1, "Generating base video...")
            output = self.pipe.generate_t2v(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )[0]

            # Save base video
            output_tensor = torch.from_numpy(np.array(output))
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            base_video_path = f"output_t2v_{seed}.mp4"
            write_video(base_video_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
            output_files.append(base_video_path)
            
            self.torch_gc()

            if enable_distill:
                progress(0.4, "Generating distilled video...")
                # Load CFG step LoRA
                cfg_step_lora_path = os.path.join(self.checkpoint_dir, 'lora/cfg_step_lora.safetensors')
                self.pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
                self.pipe.dit.enable_loras(['cfg_step_lora'])

                output_distill = self.pipe.generate_t2v(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=16,
                    use_distill=True,
                    guidance_scale=1.0,
                    generator=generator,
                )[0]
                
                self.pipe.dit.disable_all_loras()

                # Save distilled video
                output_processed_tensor = torch.from_numpy(np.array(output_distill))
                output_processed_tensor = (output_processed_tensor * 255).clamp(0, 255).to(torch.uint8)
                distill_video_path = f"output_t2v_distill_{seed}.mp4"
                write_video(distill_video_path, output_processed_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
                output_files.append(distill_video_path)

                if enable_refine:
                    progress(0.7, "Generating refined video...")
                    # Load refinement LoRA
                    refinement_lora_path = os.path.join(self.checkpoint_dir, 'lora/refinement_lora.safetensors')
                    self.pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
                    self.pipe.dit.enable_loras(['refinement_lora'])
                    self.pipe.dit.enable_bsa()

                    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
                    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
                    
                    del output_distill
                    self.torch_gc()

                    output_refine = self.pipe.generate_refine(
                        prompt=prompt,
                        stage1_video=stage1_video,
                        num_inference_steps=50,
                        generator=generator,
                    )[0]

                    self.pipe.dit.disable_all_loras()
                    self.pipe.dit.disable_bsa()

                    # Save refined video
                    output_tensor = torch.from_numpy(output_refine)
                    output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
                    refine_video_path = f"output_t2v_refine_{seed}.mp4"
                    write_video(refine_video_path, output_tensor, fps=30, video_codec="libx264", options={"crf": "10"})
                    output_files.append(refine_video_path)

            progress(1.0, "Generation complete!")
            
        except Exception as e:
            print(f"Error in text-to-video generation: {str(e)}")
            return [f"Error: {str(e)}"]
        
        return output_files

    def generate_image_to_video(
        self,
        image: PIL.Image.Image,
        prompt: str,
        negative_prompt: str = "",
        resolution: str = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 42,
        enable_distill: bool = True,
        enable_refine: bool = True,
        progress=gr.Progress()
    ) -> List[str]:
        """Image to Video generation"""
        if image is None or not prompt.strip():
            return []
        
        progress(0, "Initializing generation...")
        
        generator = torch.Generator(device=self.local_rank)
        generator.manual_seed(seed)
        
        output_files = []
        target_size = image.size  # (width, height)
        
        # Default negative prompt if empty
        if not negative_prompt.strip():
            negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        try:
            # Stage 1: Basic I2V generation
            progress(0.1, "Generating base video...")
            output = self.pipe.generate_i2v(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution=resolution,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )[0]

            # Process and save base video
            output_processed = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
            output_processed = [PIL.Image.fromarray(img) for img in output_processed]
            output_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_processed]

            output_tensor = torch.from_numpy(np.array(output_processed))
            base_video_path = f"output_i2v_{seed}.mp4"
            write_video(base_video_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
            output_files.append(base_video_path)
            
            del output
            self.torch_gc()

            if enable_distill:
                progress(0.4, "Generating distilled video...")
                # Load CFG step LoRA
                cfg_step_lora_path = os.path.join(self.checkpoint_dir, 'lora/cfg_step_lora.safetensors')
                self.pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
                self.pipe.dit.enable_loras(['cfg_step_lora'])

                output_distill = self.pipe.generate_i2v(
                    image=image,
                    prompt=prompt,
                    resolution=resolution,
                    num_frames=num_frames,
                    num_inference_steps=16,
                    use_distill=True,
                    guidance_scale=1.0,
                    generator=generator,
                )[0]
                
                self.pipe.dit.disable_all_loras()

                # Process and save distilled video
                output_processed = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
                output_processed = [PIL.Image.fromarray(img) for img in output_processed]
                output_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_processed]

                output_processed_tensor = torch.from_numpy(np.array(output_processed))
                distill_video_path = f"output_i2v_distill_{seed}.mp4"
                write_video(distill_video_path, output_processed_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
                output_files.append(distill_video_path)

                if enable_refine:
                    progress(0.7, "Generating refined video...")
                    # Load refinement LoRA
                    refinement_lora_path = os.path.join(self.checkpoint_dir, 'lora/refinement_lora.safetensors')
                    self.pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
                    self.pipe.dit.enable_loras(['refinement_lora'])
                    self.pipe.dit.enable_bsa()

                    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
                    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
                    
                    del output_distill
                    self.torch_gc()

                    output_refine = self.pipe.generate_refine(
                        image=image,
                        prompt=prompt,
                        stage1_video=stage1_video,
                        num_cond_frames=1,
                        num_inference_steps=50,
                        generator=generator,
                    )[0]

                    self.pipe.dit.disable_all_loras()
                    self.pipe.dit.disable_bsa()

                    # Process and save refined video
                    output_refine_processed = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
                    output_refine_processed = [PIL.Image.fromarray(img) for img in output_refine_processed]
                    output_refine_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_refine_processed]

                    output_tensor = torch.from_numpy(np.array(output_refine_processed))
                    refine_video_path = f"output_i2v_refine_{seed}.mp4"
                    write_video(refine_video_path, output_tensor, fps=30, video_codec="libx264", options={"crf": "10"})
                    output_files.append(refine_video_path)

            progress(1.0, "Generation complete!")
            
        except Exception as e:
            print(f"Error in image-to-video generation: {str(e)}")
            return [f"Error: {str(e)}"]
        
        return output_files

    def generate_video_continuation(
        self,
        video_file: str,
        prompt: str,
        negative_prompt: str = "",
        resolution: str = "480p",
        num_frames: int = 93,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 42,
        enable_distill: bool = True,
        enable_refine: bool = True,
        progress=gr.Progress()
    ) -> List[str]:
        """Video Continuation generation"""
        if video_file is None or not prompt.strip():
            return []
        
        progress(0, "Loading video...")
        
        try:
            video = load_video(video_file)
            target_fps = 15
            target_size = video[0].size  # (width, height)
            
            # Get original FPS
            cap = cv2.VideoCapture(video_file)
            current_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            stride = max(1, round(current_fps / target_fps))
            
            generator = torch.Generator(device=self.local_rank)
            generator.manual_seed(seed)
            
            output_files = []
            
            # Default negative prompt if empty
            if not negative_prompt.strip():
                negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
            
            # Stage 1: Basic VC generation
            progress(0.1, "Generating continuation...")
            output = self.pipe.generate_vc(
                video=video[::stride],
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution=resolution,
                num_frames=num_frames,
                num_cond_frames=num_cond_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=False,
            )[0]

            # Process and save base video
            output_processed = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
            output_processed = [PIL.Image.fromarray(img) for img in output_processed]
            output_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_processed]

            # Combine with original video
            final_output = video[::stride] + output_processed[num_cond_frames:]
            output_tensor = torch.from_numpy(np.array(final_output))
            base_video_path = f"output_vc_{seed}.mp4"
            write_video(base_video_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
            output_files.append(base_video_path)
            
            del output
            self.torch_gc()

            if enable_distill:
                progress(0.4, "Generating distilled continuation...")
                # Load CFG step LoRA
                cfg_step_lora_path = os.path.join(self.checkpoint_dir, 'lora/cfg_step_lora.safetensors')
                self.pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
                self.pipe.dit.enable_loras(['cfg_step_lora'])

                output_distill = self.pipe.generate_vc(
                    video=video[::stride],
                    prompt=prompt,
                    resolution=resolution,
                    num_frames=num_frames,
                    num_cond_frames=num_cond_frames,
                    num_inference_steps=16,
                    use_distill=True,
                    guidance_scale=1.0,
                    generator=generator,
                    use_kv_cache=True,
                    offload_kv_cache=False,
                    enhance_hf=False,
                )[0]
                
                self.pipe.dit.disable_all_loras()

                # Process and save distilled video
                output_processed = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
                output_processed = [PIL.Image.fromarray(img) for img in output_processed]
                output_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_processed]

                final_output = video[::stride] + output_processed[num_cond_frames:]
                output_tensor = torch.from_numpy(np.array(final_output))
                distill_video_path = f"output_vc_distill_{seed}.mp4"
                write_video(distill_video_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
                output_files.append(distill_video_path)

                if enable_refine:
                    progress(0.7, "Generating refined continuation...")
                    # Load refinement LoRA
                    refinement_lora_path = os.path.join(self.checkpoint_dir, 'lora/refinement_lora.safetensors')
                    self.pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
                    self.pipe.dit.enable_loras(['refinement_lora'])
                    self.pipe.dit.enable_bsa()

                    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
                    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
                    
                    del output_distill
                    self.torch_gc()

                    # Adjust stride for 30fps refinement
                    refine_fps = 30
                    refine_stride = max(1, round(current_fps / refine_fps))

                    output_refine = self.pipe.generate_refine(
                        video=video[::refine_stride],
                        prompt=prompt,
                        stage1_video=stage1_video,
                        num_cond_frames=num_cond_frames*2,
                        num_inference_steps=50,
                        generator=generator,
                    )[0]

                    self.pipe.dit.disable_all_loras()
                    self.pipe.dit.disable_bsa()

                    # Process and save refined video
                    output_refine_processed = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
                    output_refine_processed = [PIL.Image.fromarray(img) for img in output_refine_processed]
                    output_refine_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_refine_processed]
                    
                    final_output = video[::refine_stride] + output_refine_processed[num_cond_frames*2:]
                    output_tensor = torch.from_numpy(np.array(final_output))
                    refine_video_path = f"output_vc_refine_{seed}.mp4"
                    write_video(refine_video_path, output_tensor, fps=30, video_codec="libx264", options={"crf": "10"})
                    output_files.append(refine_video_path)

            progress(1.0, "Generation complete!")
            
        except Exception as e:
            print(f"Error in video continuation: {str(e)}")
            return [f"Error: {str(e)}"]
        
        return output_files

    def generate_long_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_segments: int = 5,
        num_frames: int = 93,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 42,
        enable_refine: bool = True,
        progress=gr.Progress()
    ) -> str:
        """Long Video generation"""
        if not prompt.strip():
            return None
        
        progress(0, "Initializing long video generation...")
        
        generator = torch.Generator(device=self.local_rank)
        generator.manual_seed(seed)
        
        output_files = []
        
        # Default negative prompt if empty
        if not negative_prompt.strip():
            negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        try:
            # Stage 1: Generate initial video
            progress(0.05, "Generating initial segment...")
            output = self.pipe.generate_t2v(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=480,
                width=832,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )[0]

            # Save initial segment
            output_tensor = torch.from_numpy(np.array(output))
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            initial_video_path = f"output_long_video_0_{seed}.mp4"
            write_video(initial_video_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})

            video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
            video = [PIL.Image.fromarray(img) for img in video]
            del output
            self.torch_gc()

            target_size = video[0].size
            current_video = video
            all_generated_frames = video

            # Generate additional segments
            for segment_idx in range(num_segments):
                progress_val = 0.1 + (segment_idx / num_segments) * 0.6
                progress(progress_val, f"Generating segment {segment_idx + 1}/{num_segments}...")
                
                output = self.pipe.generate_vc(
                    video=current_video,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    resolution='480p',
                    num_frames=num_frames,
                    num_cond_frames=num_cond_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    use_kv_cache=True,
                    offload_kv_cache=False,
                    enhance_hf=True
                )[0]

                new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
                new_video = [PIL.Image.fromarray(img) for img in new_video]
                new_video = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in new_video]
                del output

                all_generated_frames.extend(new_video[num_cond_frames:])
                current_video = new_video

                # Save intermediate result
                output_tensor = torch.from_numpy(np.array(all_generated_frames))
                segment_video_path = f"output_long_video_{segment_idx+1}_{seed}.mp4"
                write_video(segment_video_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})

            output_files.append(segment_video_path)  # Add final long video

            if enable_refine:
                progress(0.75, "Starting refinement process...")
                # Load refinement LoRA
                refinement_lora_path = os.path.join(self.checkpoint_dir, 'lora/refinement_lora.safetensors')
                self.pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
                self.pipe.dit.enable_loras(['refinement_lora'])
                self.pipe.dit.enable_bsa()

                self.torch_gc()
                cur_condition_video = None
                cur_num_cond_frames = 0
                start_id = 0
                all_refine_frames = []

                for segment_idx in range(num_segments + 1):
                    refine_progress = 0.75 + (segment_idx / (num_segments + 1)) * 0.25
                    progress(refine_progress, f"Refining segment {segment_idx + 1}/{num_segments + 1}...")

                    output_refine = self.pipe.generate_refine(
                        video=cur_condition_video,
                        prompt='',
                        stage1_video=all_generated_frames[start_id:start_id+num_frames],
                        num_cond_frames=cur_num_cond_frames,
                        num_inference_steps=50,
                        generator=generator,
                    )[0]

                    new_video = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
                    new_video = [PIL.Image.fromarray(img) for img in new_video]
                    del output_refine

                    all_refine_frames.extend(new_video[cur_num_cond_frames:])
                    cur_condition_video = new_video
                    cur_num_cond_frames = num_cond_frames * 2
                    start_id = start_id + num_frames - num_cond_frames

                # Save final refined video
                output_tensor = torch.from_numpy(np.array(all_refine_frames))
                refine_video_path = f"output_longvideo_refine_{seed}.mp4"
                write_video(refine_video_path, output_tensor, fps=30, video_codec="libx264", options={"crf": "10"})
                output_files.append(refine_video_path)

                self.pipe.dit.disable_all_loras()
                self.pipe.dit.disable_bsa()

            progress(1.0, "Long video generation complete!")
            
        except Exception as e:
            print(f"Error in long video generation: {str(e)}")
            return None
        
        # Return the final video (refined if available, otherwise the last segment)
        if output_files:
            return output_files[-1]  # Return the last (most complete) video
        else:
            return None


def create_webui(checkpoint_dir: str, enable_compile: bool = True):
    """Create Gradio WebUI"""
    
    # Initialize the LongCat Video WebUI
    webui = LongCatVideoWebUI(checkpoint_dir, enable_compile)
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .video-output {
        max-height: 400px;
    }
    """
    
    with gr.Blocks(css=css, title="LongCat-Video 网页界面") as demo:
        gr.Markdown("# 🎬 LongCat-Video 网页界面")
        gr.Markdown("使用 LongCat-Video 模型生成高质量视频")
        
        with gr.Tabs():
            # Tab 1: Text to Video
            with gr.TabItem("📝 文本生成视频"):
                with gr.Row():
                    with gr.Column():
                        t2v_prompt = gr.Textbox(
                            label="提示词", 
                            placeholder="描述您想要生成的视频内容...",
                            lines=3,
                            value="In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."
                        )
                        t2v_negative = gr.Textbox(
                            label="负向提示词 (可选)", 
                            placeholder="描述您不希望在视频中出现的内容...",
                            lines=2
                        )
                        
                        with gr.Row():
                            t2v_width = gr.Slider(256, 1024, value=832, step=64, label="宽度")
                            t2v_height = gr.Slider(256, 1024, value=480, step=64, label="高度")
                        
                        with gr.Row():
                            t2v_frames = gr.Slider(1, 200, value=93, step=1, label="帧数")
                            t2v_steps = gr.Slider(10, 100, value=50, step=1, label="推理步数")
                        
                        with gr.Row():
                            t2v_guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="引导强度")
                            t2v_seed = gr.Number(label="随机种子", value=42)
                        
                        with gr.Row():
                            t2v_distill = gr.Checkbox(label="启用蒸馏", value=True)
                            t2v_refine = gr.Checkbox(label="启用精化", value=True)
                        
                        t2v_button = gr.Button("🎬 生成视频", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### 生成结果")
                        t2v_basic = gr.Video(label="基础版本", show_label=True)
                        t2v_distill_out = gr.Video(label="蒸馏版本 (快速)", show_label=True) 
                        t2v_refine_out = gr.Video(label="精化版本 (高质量)", show_label=True)
                
                t2v_button.click(
                    webui.generate_text_to_video,
                    inputs=[t2v_prompt, t2v_negative, t2v_height, t2v_width, t2v_frames, t2v_steps, t2v_guidance, t2v_seed, t2v_distill, t2v_refine],
                    outputs=[t2v_basic, t2v_distill_out, t2v_refine_out]
                )
            
            # Tab 2: Image to Video
            with gr.TabItem("🖼️ 图片生成视频"):
                with gr.Row():
                    with gr.Column():
                        i2v_image = gr.Image(label="输入图片", type="pil")
                        
                        # 示例图片
                        gr.Examples(
                            examples=["assets/girl.png"],
                            inputs=[i2v_image],
                            label="示例图片"
                        )
                        
                        i2v_prompt = gr.Textbox(
                            label="提示词", 
                            placeholder="描述图片应该如何动画化...",
                            lines=3,
                            value="A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip. After drinking, she places the cup back on the table and looks out the window, enjoying the peaceful atmosphere."
                        )
                        i2v_negative = gr.Textbox(
                            label="负向提示词 (可选)", 
                            placeholder="描述您不希望在视频中出现的内容...",
                            lines=2
                        )
                        
                        with gr.Row():
                            i2v_resolution = gr.Dropdown(choices=["480p", "720p"], value="480p", label="分辨率")
                            i2v_frames = gr.Slider(1, 200, value=93, step=1, label="帧数")
                        
                        with gr.Row():
                            i2v_steps = gr.Slider(10, 100, value=50, step=1, label="推理步数")
                            i2v_guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="引导强度")
                        
                        with gr.Row():
                            i2v_seed = gr.Number(label="随机种子", value=42)
                            i2v_distill = gr.Checkbox(label="启用蒸馏", value=True)
                            i2v_refine = gr.Checkbox(label="启用精化", value=True)
                        
                        i2v_button = gr.Button("🎬 生成视频", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### 生成结果")
                        i2v_basic = gr.Video(label="基础版本", show_label=True)
                        i2v_distill_out = gr.Video(label="蒸馏版本 (快速)", show_label=True)
                        i2v_refine_out = gr.Video(label="精化版本 (高质量)", show_label=True)
                
                i2v_button.click(
                    webui.generate_image_to_video,
                    inputs=[i2v_image, i2v_prompt, i2v_negative, i2v_resolution, i2v_frames, i2v_steps, i2v_guidance, i2v_seed, i2v_distill, i2v_refine],
                    outputs=[i2v_basic, i2v_distill_out, i2v_refine_out]
                )
            
            # Tab 3: Video Continuation
            with gr.TabItem("🔄 视频续接"):
                with gr.Row():
                    with gr.Column():
                        vc_video = gr.Video(label="输入视频")
                        
                        # 示例视频
                        gr.Examples(
                            examples=["assets/motorcycle.mp4"],
                            inputs=[vc_video],
                            label="示例视频"
                        )
                        
                        vc_prompt = gr.Textbox(
                            label="提示词", 
                            placeholder="描述视频应该如何继续...",
                            lines=3,
                            value="A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides. The video captures the journey from the rider's perspective, emphasizing the sense of motion and adventure."
                        )
                        vc_negative = gr.Textbox(
                            label="负向提示词 (可选)", 
                            placeholder="描述您不希望在续接中出现的内容...",
                            lines=2
                        )
                        
                        with gr.Row():
                            vc_resolution = gr.Dropdown(choices=["480p", "720p"], value="480p", label="分辨率")
                            vc_frames = gr.Slider(1, 200, value=93, step=1, label="帧数")
                        
                        with gr.Row():
                            vc_cond_frames = gr.Slider(1, 50, value=13, step=1, label="条件帧数")
                            vc_steps = gr.Slider(10, 100, value=50, step=1, label="推理步数")
                        
                        with gr.Row():
                            vc_guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="引导强度")
                            vc_seed = gr.Number(label="随机种子", value=42)
                        
                        with gr.Row():
                            vc_distill = gr.Checkbox(label="启用蒸馏", value=True)
                            vc_refine = gr.Checkbox(label="启用精化", value=True)
                        
                        vc_button = gr.Button("🎬 续接视频", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### 生成结果")
                        vc_basic = gr.Video(label="基础版本", show_label=True)
                        vc_distill_out = gr.Video(label="蒸馏版本 (快速)", show_label=True)
                        vc_refine_out = gr.Video(label="精化版本 (高质量)", show_label=True)
                
                vc_button.click(
                    webui.generate_video_continuation,
                    inputs=[vc_video, vc_prompt, vc_negative, vc_resolution, vc_frames, vc_cond_frames, vc_steps, vc_guidance, vc_seed, vc_distill, vc_refine],
                    outputs=[vc_basic, vc_distill_out, vc_refine_out]
                )
            
            # Tab 4: Long Video
            with gr.TabItem("⏰ 长视频生成"):
                with gr.Row():
                    with gr.Column():
                        lv_prompt = gr.Textbox(
                            label="提示词", 
                            placeholder="描述长视频的叙事内容...",
                            lines=4,
                            value="realistic filming style, a person wearing a dark helmet, a deep-colored jacket, blue jeans, and bright yellow shoes rides a skateboard along a winding mountain road. The skateboarder starts in a standing position, then gradually lowers into a crouch, extending one hand to touch the road surface while maintaining a low center of gravity to navigate a sharp curve. After completing the turn, the skateboarder rises back to a standing position and continues gliding forward. The background features lush green hills flanking both sides of the road, with distant snow-capped mountain peaks rising against a clear, bright blue sky. The camera follows closely from behind, smoothly tracking the skateboarder's movements and capturing the dynamic scenery along the route. The scene is shot in natural daylight, highlighting the vivid outdoor environment and the skateboarder's fluid actions."
                        )
                        lv_negative = gr.Textbox(
                            label="负向提示词 (可选)", 
                            placeholder="描述您不希望在视频中出现的内容...",
                            lines=2
                        )
                        
                        with gr.Row():
                            lv_segments = gr.Slider(1, 20, value=5, step=1, label="视频段数")
                            lv_frames = gr.Slider(50, 200, value=93, step=1, label="每段帧数")
                        
                        with gr.Row():
                            lv_cond_frames = gr.Slider(1, 50, value=13, step=1, label="条件帧数")
                            lv_steps = gr.Slider(10, 100, value=50, step=1, label="推理步数")
                        
                        with gr.Row():
                            lv_guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="引导强度")
                            lv_seed = gr.Number(label="随机种子", value=42)
                        
                        lv_refine = gr.Checkbox(label="启用精化", value=True)
                        
                        lv_button = gr.Button("🎬 生成长视频", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### 生成结果")
                        lv_output = gr.Video(label="长视频", show_label=True)
                        gr.Markdown("⚠️ **注意**: 长视频生成可能需要大量时间和GPU内存。")
                
                lv_button.click(
                    webui.generate_long_video,
                    inputs=[lv_prompt, lv_negative, lv_segments, lv_frames, lv_cond_frames, lv_steps, lv_guidance, lv_seed, lv_refine],
                    outputs=[lv_output]
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Tips**: 
        - Use descriptive prompts for better results
        - Higher inference steps = better quality but slower generation
        - Enable refinement for highest quality output
        - Long videos require significant GPU memory and time
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video WebUI")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/LongCat-Video",
        help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--enable_compile",
        action="store_true",
        help="Enable torch compilation for faster inference"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host address"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    # Create and launch the WebUI
    demo = create_webui(args.checkpoint_dir, args.enable_compile)
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True
    )


if __name__ == "__main__":
    main()