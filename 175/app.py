import gradio as gr
import os
import json
import subprocess
import time
import yaml
from datetime import datetime
from PIL import Image
import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom dataset class to avoid modifying existing files
class InlineCustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='txt',
                 random_ratio=False, caption_dropout_rate=0.1):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        print(f'Found {len(self.images)} images in {img_dir}')
    
    def __len__(self):
        return 999999  # Large number for continuous training
    
    def __getitem__(self, idx):
        try:
            idx = random.randint(0, len(self.images) - 1)
            
            # Load and process image
            img = Image.open(self.images[idx]).convert('RGB')
            
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = self._crop_to_aspect_ratio(img, ratio)
            
            img = self._image_resize(img, self.img_size)
            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            
            # Load text caption with fixed path handling
            img_path = self.images[idx]
            txt_path = os.path.splitext(img_path)[0] + '.' + self.caption_type
            
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            # Random caption dropout
            if random.random() < self.caption_dropout_rate:
                return img, " "
            else:
                return img, prompt
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return self.__getitem__(random.randint(0, len(self.images) - 1))
    
    def _image_resize(self, image, target_size):
        """Resize image maintaining aspect ratio"""
        width, height = image.size
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def _crop_to_aspect_ratio(self, image, target_ratio):
        """Crop image to target aspect ratio"""
        width, height = image.size
        
        if target_ratio == "16:9":
            target_ratio_value = 16 / 9
        elif target_ratio == "1:1":
            target_ratio_value = 1
        elif target_ratio == "4:3":
            target_ratio_value = 4 / 3
        else:
            return image
        
        current_ratio = width / height
        
        if current_ratio > target_ratio_value:
            # Image is too wide, crop from sides
            new_width = int(height * target_ratio_value)
            offset = (width - new_width) // 2
            crop_box = (offset, 0, offset + new_width, height)
        else:
            # Image is too tall, crop from top/bottom
            new_height = int(width / target_ratio_value)
            offset = (height - new_height) // 2
            crop_box = (0, offset, width, offset + new_height)
        
        return image.crop(crop_box)

def inline_data_loader(img_dir, img_size, caption_type, random_ratio, caption_dropout_rate, train_batch_size, num_workers):
    """Custom data loader function to replace the original loader"""
    dataset = InlineCustomImageDataset(
        img_dir=img_dir,
        img_size=img_size,
        caption_type=caption_type,
        random_ratio=random_ratio,
        caption_dropout_rate=caption_dropout_rate
    )
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

class LoRATrainerWebUI:
    def __init__(self):
        self.dataset_dir = "./datasets"
        self.training_data_dir = "./my_training_data"
        self.checkpoints_dir = "./checkpoints"
        self.config_dir = "./train_configs"
        self.output_dir = "./output"
        
        # Create directories if they don't exist
        for dir_path in [self.dataset_dir, self.training_data_dir, self.checkpoints_dir, self.config_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def download_dataset(self, dataset_name, split, num_samples, progress=gr.Progress()):
        """下载并准备数据集"""
        try:
            progress(0, desc="正在下载数据集...")
            
            # Download dataset
            dataset = load_dataset(dataset_name, split=split, streaming=False)
            
            if num_samples > 0:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            progress(0.3, desc="正在处理图像...")
            
            # Save images and create text descriptions
            image_count = 0
            total_images = len(dataset)
            
            for i, item in enumerate(dataset):
                if 'image' in item:
                    # Save image
                    image_filename = f"image_{i+1:03d}.png"
                    image_path = os.path.join(self.training_data_dir, image_filename)
                    item['image'].save(image_path)
                    
                    # Create text description from filename or metadata
                    text_filename = f"image_{i+1:03d}.txt"
                    text_path = os.path.join(self.training_data_dir, text_filename)
                    
                    # Extract description from filename or create one
                    if dataset_name == "monadical-labs/minecraft-preview":
                        # Extract character name from image filename
                        original_filename = item.get('filename', f'character_{i+1}')
                        char_name = original_filename.replace('.png', '').replace('_', ' ')
                        description = f"A minecraft character: {char_name}, minecraft style, pixelated, blocky, game character"
                    else:
                        description = f"An image from {dataset_name}, detailed, high quality"
                    
                    with open(text_path, 'w') as f:
                        f.write(description)
                    
                    image_count += 1
                    progress((i + 1) / total_images * 0.7 + 0.3, desc=f"已处理 {i+1}/{total_images} 张图像")
            
            progress(1.0, desc="数据集准备完成！")
            return f"成功下载并准备了 {image_count} 张图像，来自 {dataset_name}！\n图像保存位置：{self.training_data_dir}"
        
        except Exception as e:
            return f"数据集下载错误：{str(e)}"
    
    def validate_dataset(self):
        """验证准备好的数据集"""
        try:
            if not os.path.exists(self.training_data_dir):
                return "❌ 未找到训练数据目录。请先下载数据集。"
            
            image_files = [f for f in os.listdir(self.training_data_dir) if f.endswith('.png')]
            text_files = [f for f in os.listdir(self.training_data_dir) if f.endswith('.txt')]
            
            image_count = len(image_files)
            text_count = len(text_files)
            
            # Check for paired files
            paired_count = 0
            paired_files = []
            for img_file in image_files:
                txt_file = img_file.replace('.png', '.txt')
                if txt_file in text_files:
                    paired_count += 1
                    paired_files.append((img_file, txt_file))
            
            # Sample a few text descriptions
            sample_descriptions = []
            for i, (img_file, txt_file) in enumerate(paired_files[:3]):
                try:
                    with open(os.path.join(self.training_data_dir, txt_file), 'r') as f:
                        desc = f.read().strip()
                        sample_descriptions.append(f"{img_file}: {desc}")
                except:
                    continue
            
            validation_result = f"""
📊 数据集验证结果：
• 发现图像：{image_count} 张
• 发现文本文件：{text_count} 个  
• 正确配对：{paired_count} 对
• 数据集目录：{self.training_data_dir}

📝 示例描述：
{chr(10).join(sample_descriptions)}

{'✅ 数据集已准备好进行训练！' if paired_count > 0 else '❌ 未找到有效的图像-文本配对！'}
"""
            return validation_result
        
        except Exception as e:
            return f"❌ 数据集验证错误：{str(e)}"
    
    def get_dataset_status(self):
        """获取当前数据集状态"""
        try:
            if not os.path.exists(self.training_data_dir):
                return "❌ 未找到数据集"
            
            image_files = [f for f in os.listdir(self.training_data_dir) if f.endswith('.png')]
            text_files = [f for f in os.listdir(self.training_data_dir) if f.endswith('.txt')]
            
            if len(image_files) > 0 and len(text_files) > 0:
                return f"✅ {len(image_files)} 张图像已准备好"
            else:
                return "⚠️ 数据集不完整"
        except:
            return "❌ 数据集错误"
    
    def get_dataset_preview(self, num_samples=5):
        """获取数据集预览，包含图像和标题"""
        try:
            if not os.path.exists(self.training_data_dir):
                return [], "❌ 未找到数据集"
            
            image_files = [f for f in os.listdir(self.training_data_dir) if f.endswith('.png')]
            image_files.sort()  # Sort for consistent ordering
            
            preview_images = []
            preview_captions = []
            
            # Get first few samples for preview
            for i, img_file in enumerate(image_files[:num_samples]):
                img_path = os.path.join(self.training_data_dir, img_file)
                txt_file = img_file.replace('.png', '.txt')
                txt_path = os.path.join(self.training_data_dir, txt_file)
                
                # Load image
                if os.path.exists(img_path):
                    preview_images.append(img_path)
                
                # Load caption
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        caption = f.read().strip()
                        preview_captions.append(f"#{i+1}: {caption}")
                else:
                    preview_captions.append(f"#{i+1}: 未找到标题")
            
            preview_text = "\n\n".join(preview_captions)
            return preview_images, preview_text
        
        except Exception as e:
            return [], f"❌ 加载预览时出错：{str(e)}"
    
    def refresh_preview(self):
        """刷新数据集预览"""
        return self.get_dataset_preview()
    
    def start_training(self, model_name, learning_rate, batch_size, epochs, resolution, 
                      lora_rank, lora_alpha, gradient_accumulation_steps, save_steps, logging_steps, 
                      progress=gr.Progress()):
        """直接启动 LoRA 训练（完全内嵌，无需配置文件）"""
        try:
            import copy
            from copy import deepcopy
            from accelerate import Accelerator
            from accelerate.utils import ProjectConfiguration
            import datasets
            import diffusers
            from diffusers import FlowMatchEulerDiscreteScheduler
            from diffusers import (
                AutoencoderKLQwenImage,
                QwenImagePipeline,
                QwenImageTransformer2DModel,
            )
            from diffusers.optimization import get_scheduler
            from diffusers.training_utils import (
                compute_density_for_timestep_sampling,
                compute_loss_weighting_for_sd3,
            )
            from diffusers.utils import convert_state_dict_to_diffusers
            from diffusers.utils.torch_utils import is_compiled_module
            from peft import LoraConfig
            from peft.utils import get_peft_model_state_dict
            import transformers
            from tqdm.auto import tqdm
            
            progress(0, desc="准备训练环境...")
            
            # Validate data directory
            if not os.path.exists(self.training_data_dir):
                return "❌ 训练数据目录不存在，请先下载数据集！"
            
            # Validate model directory
            if not os.path.exists(model_name):
                return f"❌ 模型路径不存在：{model_name}"
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            logging_dir = os.path.join(self.output_dir, "logs")
            
            progress(0.1, desc="初始化训练设置...")
            
            # Calculate max steps
            image_files = [f for f in os.listdir(self.training_data_dir) if f.endswith('.png')]
            dataset_size = len(image_files)
            max_train_steps = max(100, (dataset_size // batch_size) * epochs)
            
            # Initialize accelerator with bf16 to avoid gradient scaling issues
            accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir, logging_dir=logging_dir)
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision="bf16",  # Use bf16 instead of fp16 to avoid gradient scaling issues
                project_config=accelerator_project_config,
            )
            
            def unwrap_model(model):
                model = accelerator.unwrap_model(model)
                model = model._orig_mod if is_compiled_module(model) else model
                return model
            
            # Set logging
            if accelerator.is_local_main_process:
                datasets.utils.logging.set_verbosity_warning()
                transformers.utils.logging.set_verbosity_warning()
                diffusers.utils.logging.set_verbosity_info()
            else:
                datasets.utils.logging.set_verbosity_error()
                transformers.utils.logging.set_verbosity_error()
                diffusers.utils.logging.set_verbosity_error()
            
            progress(0.2, desc="加载模型...")
            
            # Set dtype based on mixed precision
            weight_dtype = torch.float32
            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
            
            # Load models
            text_encoding_pipeline = QwenImagePipeline.from_pretrained(
                model_name, transformer=None, vae=None, torch_dtype=weight_dtype
            )
            vae = AutoencoderKLQwenImage.from_pretrained(
                model_name,
                subfolder="vae",
            )
            flux_transformer = QwenImageTransformer2DModel.from_pretrained(
                model_name,
                subfolder="transformer",
            )
            
            progress(0.3, desc="配置 LoRA...")
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            
            noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_name,
                subfolder="scheduler",
            )
            
            flux_transformer.to(accelerator.device, dtype=weight_dtype)
            flux_transformer.add_adapter(lora_config)
            text_encoding_pipeline.to(accelerator.device)
            noise_scheduler_copy = copy.deepcopy(noise_scheduler)
            
            def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
                schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
                timesteps = timesteps.to(accelerator.device)
                step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
                sigma = sigmas[step_indices].flatten()
                while len(sigma.shape) < n_dim:
                    sigma = sigma.unsqueeze(-1)
                return sigma
            
            # Freeze base model parameters
            vae.requires_grad_(False)
            flux_transformer.requires_grad_(False)
            flux_transformer.train()
            
            progress(0.4, desc="配置优化器...")
            
            # Set trainable parameters
            trainable_params = []
            for n, param in flux_transformer.named_parameters():
                if 'lora' not in n:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    trainable_params.append(param)
            
            print(f"Trainable parameters: {sum([p.numel() for p in trainable_params]) / 1000000:.1f}M")
            
            flux_transformer.enable_gradient_checkpointing()
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                eps=1e-8,
            )
            
            progress(0.5, desc="准备数据...")
            
            # Data configuration - using inline data loader  
            train_dataloader = inline_data_loader(
                img_dir=self.training_data_dir,
                img_size=resolution,
                caption_type='txt',
                random_ratio=False,
                caption_dropout_rate=0.1,
                train_batch_size=batch_size,
                num_workers=1
            )
            
            lr_scheduler = get_scheduler(
                "constant",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=max_train_steps * accelerator.num_processes,
            )
            
            progress(0.6, desc="准备训练...")
            
            # Prepare for training
            vae.to(accelerator.device, dtype=weight_dtype)
            flux_transformer, optimizer, _, lr_scheduler = accelerator.prepare(
                flux_transformer, optimizer, deepcopy(train_dataloader), lr_scheduler
            )
            
            global_step = 0
            total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
            
            print(f"***** Running training *****")
            print(f"  Instantaneous batch size per device = {batch_size}")
            print(f"  Total train batch size = {total_batch_size}")
            print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
            print(f"  Max training steps = {max_train_steps}")
            
            progress(0.7, desc="开始训练...")
            
            vae_scale_factor = 2 ** len(vae.temperal_downsample)
            
            for epoch in range(1):
                train_loss = 0.0
                step_count = 0
                
                for step, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(flux_transformer):
                        img, prompts = batch
                        
                        with torch.no_grad():
                            pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                            pixel_values = pixel_values.unsqueeze(2)
                            
                            pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                            pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                            
                            latents_mean = (
                                torch.tensor(vae.config.latents_mean)
                                .view(1, 1, vae.config.z_dim, 1, 1)
                                .to(pixel_latents.device, pixel_latents.dtype)
                            )
                            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                                pixel_latents.device, pixel_latents.dtype
                            )
                            pixel_latents = (pixel_latents - latents_mean) * latents_std
                            
                            bsz = pixel_latents.shape[0]
                            noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                            u = compute_density_for_timestep_sampling(
                                weighting_scheme="none",
                                batch_size=bsz,
                                logit_mean=0.0,
                                logit_std=1.0,
                                mode_scale=1.29,
                            )
                            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                            timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
                            
                            sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                            
                            # Pack the latents
                            packed_noisy_model_input = QwenImagePipeline._pack_latents(
                                noisy_model_input,
                                bsz, 
                                noisy_model_input.shape[2],
                                noisy_model_input.shape[3],
                                noisy_model_input.shape[4],
                            )
                            
                            # Encode prompts
                            img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz
                            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                                prompt=prompts,
                                device=packed_noisy_model_input.device,
                                num_images_per_prompt=1,
                                max_sequence_length=1024,
                            )
                            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                        
                        # Model prediction
                        model_pred = flux_transformer(
                            hidden_states=packed_noisy_model_input,
                            timestep=timesteps / 1000,
                            guidance=None,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            return_dict=False,
                        )[0]
                        
                        model_pred = QwenImagePipeline._unpack_latents(
                            model_pred,
                            height=noisy_model_input.shape[3] * vae_scale_factor,
                            width=noisy_model_input.shape[4] * vae_scale_factor,
                            vae_scale_factor=vae_scale_factor,
                        )
                        
                        # Compute loss
                        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                        target = noise - pixel_latents
                        target = target.permute(0, 2, 1, 3, 4)
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                            1,
                        )
                        loss = loss.mean()
                        
                        avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                        train_loss += avg_loss.item() / gradient_accumulation_steps
                        
                        # Backpropagate
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(flux_transformer.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    
                    # Update progress and save checkpoints
                    if accelerator.sync_gradients:
                        global_step += 1
                        step_count += 1
                        train_loss = 0.0
                        
                        # Update progress
                        training_progress = 0.7 + (global_step / max_train_steps) * 0.25  # 70% to 95%
                        progress(training_progress, desc=f"训练中... 步骤 {global_step}/{max_train_steps}")
                        
                        if global_step % save_steps == 0:
                            if accelerator.is_main_process:
                                save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                                os.makedirs(save_path, exist_ok=True)
                                
                                unwrapped_flux_transformer = unwrap_model(flux_transformer)
                                flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                                    get_peft_model_state_dict(unwrapped_flux_transformer)
                                )
                                
                                QwenImagePipeline.save_lora_weights(
                                    save_path,
                                    flux_transformer_lora_state_dict,
                                    safe_serialization=True,
                                )
                                
                                print(f"Saved checkpoint to {save_path}")
                        
                        if global_step >= max_train_steps:
                            break
                    
                    if global_step >= max_train_steps:
                        break
            
            accelerator.wait_for_everyone()
            accelerator.end_training()
            
            progress(1.0, desc="训练完成！")
            return f"✅ 训练成功完成！\n\n训练步数: {global_step}\n输出目录: {self.output_dir}"
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 训练时出错：{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg
    
    def load_inference_model(self, model_path):
        """加载推理模型"""
        try:
            self.pipe = None
            if os.path.exists(model_path):
                # 设置设备和数据类型
                if torch.cuda.is_available():
                    torch_dtype = torch.bfloat16
                    device = "cuda"
                    device_info = f"GPU: {torch.cuda.get_device_name()}"
                else:
                    torch_dtype = torch.float32
                    device = "cpu"
                    device_info = "CPU"
                
                # 加载基础模型 - 使用正确的 DiffusionPipeline
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe = self.pipe.to(device)
                
                # 查找 LoRA 权重文件
                lora_path = os.path.join(model_path, "pytorch_lora_weights.safetensors")
                if not os.path.exists(lora_path):
                    # 尝试在子目录中查找
                    lora_path = os.path.join(model_path, "lora", "pytorch_lora_weights.safetensors")
                
                # 加载 LoRA 权重
                if os.path.exists(lora_path):
                    self.pipe.load_lora_weights(lora_path, adapter_name="minecraft_lora")
                    return f"✅ 模型和 LoRA 加载成功！\n设备：{device_info}\nLoRA 路径：{lora_path}"
                else:
                    return f"✅ 基础模型加载成功！\n设备：{device_info}\n⚠️ 未找到 LoRA 权重文件"
            else:
                return f"❌ 未找到模型路径：{model_path}"
        
        except Exception as e:
            return f"❌ 加载模型时出错：{str(e)}"
    
    def generate_image(self, prompt, negative_prompt, steps, cfg_scale, seed, width, height, progress=gr.Progress()):
        """使用加载的模型生成图像"""
        try:
            if not hasattr(self, 'pipe') or self.pipe is None:
                return None, "❌ 请先加载模型！"
            
            progress(0, desc="正在生成图像...")
            
            # 设置随机种子
            if seed < 0:
                seed = torch.randint(0, 1000000, (1,)).item()
            
            # 获取设备信息
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
            
            progress(0.2, desc="正在推理...")
            
            # 生成图像 - 使用与 inference.py 相同的参数
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg_scale),
                generator=generator
            )
            
            progress(0.8, desc="正在保存图像...")
            
            # 保存生成的图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"generated_{timestamp}_seed_{seed}.png")
            result.images[0].save(output_path)
            
            progress(1.0, desc="图像生成完成！")
            
            generation_info = f"""✅ 图像生成成功！
保存路径：{output_path}
提示词：{prompt}
负面提示词：{negative_prompt}
尺寸：{width}x{height}
推理步数：{steps}
CFG Scale：{cfg_scale}
随机种子：{seed}"""
            
            return result.images[0], generation_info
        
        except Exception as e:
            return None, f"❌ 生成图像时出错：{str(e)}"
    
    def get_sample_prompts(self):
        """获取 Minecraft 角色的示例提示词"""
        prompts = [
            "A powerful wizard in Minecraft style, magical staff, blocky robes, pixelated, voxel art, fantasy theme",
            "A brave knight in Minecraft style, iron armor, diamond sword, blocky design, pixelated, voxel art, medieval theme",
            "A fire mage in Minecraft style, flame staff, orange robes, blocky fire effects, pixelated, voxel art, magic theme",
            "An ice warrior in Minecraft style, frost armor, ice sword, blocky ice crystals, pixelated, voxel art, winter theme",
            "A forest archer in Minecraft style, wooden bow, green cloak, blocky trees background, pixelated, voxel art, nature theme"
        ]
        return prompts

# Initialize the WebUI class
webui = LoRATrainerWebUI()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="LoRA 训练器 WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎮 LoRA 训练器 WebUI")
        gr.Markdown("完整的数据集准备、LoRA训练和推理工具包")
        
        with gr.Tabs():
            # Tab 1: Dataset Download
            with gr.TabItem("📥 数据集下载"):
                gr.Markdown("## 下载和准备训练数据集")
                
                # Add dataset status display
                with gr.Row():
                    dataset_status = gr.Textbox(
                        label="当前数据集状态",
                        value=webui.get_dataset_status(),
                        interactive=False
                    )
                    refresh_status_btn = gr.Button("🔄 刷新状态")
                
                with gr.Row():
                    with gr.Column():
                        dataset_name = gr.Textbox(
                            label="数据集名称",
                            value="monadical-labs/minecraft-preview",
                            placeholder="例如：monadical-labs/minecraft-preview"
                        )
                        split = gr.Dropdown(
                            choices=["train", "test", "validation"],
                            value="train",
                            label="数据集分割"
                        )
                        num_samples = gr.Number(
                            label="样本数量 (0 = 全部)",
                            value=100,
                            minimum=0
                        )
                        
                        download_btn = gr.Button("📥 下载数据集", variant="primary")
                        validate_btn = gr.Button("🔍 验证数据集")
                    
                    with gr.Column():
                        download_output = gr.Textbox(
                            label="下载状态",
                            lines=10,
                            interactive=False
                        )
                
                download_btn.click(
                    webui.download_dataset,
                    inputs=[dataset_name, split, num_samples],
                    outputs=[download_output]
                )
                
                validate_btn.click(
                    webui.validate_dataset,
                    outputs=[download_output]
                )
                
                refresh_status_btn.click(
                    webui.get_dataset_status,
                    outputs=[dataset_status]
                )
                
                # Add dataset preview section
                gr.Markdown("## 数据集预览")
                with gr.Row():
                    with gr.Column():
                        preview_btn = gr.Button("👁️ 预览数据集")
                        preview_gallery = gr.Gallery(
                            label="示例图像",
                            show_label=True,
                            elem_id="preview_gallery",
                            columns=3,
                            rows=2,
                            height="auto"
                        )
                    
                    with gr.Column():
                        preview_captions = gr.Textbox(
                            label="示例标题",
                            lines=10,
                            interactive=False,
                            placeholder="点击'预览数据集'查看示例标题..."
                        )
                
                preview_btn.click(
                    webui.get_dataset_preview,
                    outputs=[preview_gallery, preview_captions]
                )
            
            # Tab 2: Training
            with gr.TabItem("🚀 训练"):
                gr.Markdown("## LoRA 训练配置")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 基础配置")
                        model_name = gr.Textbox(
                            label="基础模型路径",
                            value="./checkpoints/Qwen-Image",
                            placeholder="基础模型路径"
                        )
                        learning_rate = gr.Number(
                            label="学习率",
                            value=1e-4,
                            minimum=1e-6,
                            maximum=1e-2,
                            step=1e-5
                        )
                        batch_size = gr.Number(
                            label="批量大小",
                            value=1,
                            minimum=1,
                            maximum=8
                        )
                        epochs = gr.Number(
                            label="训练轮数",
                            value=10,
                            minimum=1,
                            maximum=100
                        )
                        resolution = gr.Dropdown(
                            choices=[512, 768, 1024],
                            value=1024,
                            label="训练分辨率"
                        )
                        
                        gr.Markdown("### LoRA 配置")
                        lora_rank = gr.Number(
                            label="LoRA 秩",
                            value=32,
                            minimum=1,
                            maximum=128
                        )
                        lora_alpha = gr.Number(
                            label="LoRA Alpha",
                            value=16,
                            minimum=1,
                            maximum=64
                        )
                        
                        gr.Markdown("### 高级设置")
                        gradient_accumulation_steps = gr.Number(
                            label="梯度累积步数",
                            value=1,
                            minimum=1,
                            maximum=8
                        )
                        save_steps = gr.Number(
                            label="保存步数",
                            value=500,
                            minimum=100,
                            maximum=2000
                        )
                        logging_steps = gr.Number(
                            label="日志步数",
                            value=50,
                            minimum=10,
                            maximum=200
                        )
                        
                        train_btn = gr.Button("🚀 开始训练", variant="primary")
                    
                    with gr.Column():
                        training_output = gr.Textbox(
                            label="训练日志",
                            lines=20,
                            interactive=False
                        )
                
                train_btn.click(
                    webui.start_training,
                    inputs=[model_name, learning_rate, batch_size, epochs, resolution, 
                            lora_rank, lora_alpha, gradient_accumulation_steps, save_steps, logging_steps],
                    outputs=[training_output]
                )
            
            # Tab 3: Inference
            with gr.TabItem("🎨 推理"):
                gr.Markdown("## 使用训练好的 LoRA 生成图像")
                
                with gr.Row():
                    with gr.Column():
                        model_path = gr.Textbox(
                            label="模型路径",
                            value="./checkpoints/Qwen-Image",
                            placeholder="训练好的模型路径"
                        )
                        load_model_btn = gr.Button("📂 加载模型")
                        model_status = gr.Textbox(
                            label="模型状态",
                            interactive=False
                        )
                        
                        gr.Markdown("### 生成设置")
                        prompt = gr.Textbox(
                            label="提示词",
                            lines=3,
                            placeholder="描述你想要生成的内容..."
                        )
                        
                        sample_prompts = gr.Dropdown(
                            choices=webui.get_sample_prompts(),
                            label="示例提示词",
                            interactive=True
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="负面提示词",
                            value="blurry, low quality, distorted, ugly, bad anatomy",
                            lines=2
                        )
                        
                        with gr.Row():
                            steps = gr.Slider(
                                label="推理步数",
                                minimum=10,
                                maximum=100,
                                value=30,
                                step=1
                            )
                            cfg_scale = gr.Slider(
                                label="CFG 比例",
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5
                            )
                        
                        with gr.Row():
                            seed = gr.Number(
                                label="随机种子 (-1 为随机)",
                                value=47,
                                minimum=-1
                            )
                            width = gr.Dropdown(
                                choices=[512, 768, 1024],
                                value=1024,
                                label="宽度"
                            )
                            height = gr.Dropdown(
                                choices=[512, 768, 1024],
                                value=1024,
                                label="高度"
                            )
                        
                        generate_btn = gr.Button("🎨 生成图像", variant="primary")
                    
                    with gr.Column():
                        generated_image = gr.Image(
                            label="生成的图像",
                            height=400
                        )
                        generation_status = gr.Textbox(
                            label="生成状态",
                            interactive=False
                        )
                
                # Event handlers
                load_model_btn.click(
                    webui.load_inference_model,
                    inputs=[model_path],
                    outputs=[model_status]
                )
                
                sample_prompts.change(
                    lambda x: x,
                    inputs=[sample_prompts],
                    outputs=[prompt]
                )
                
                generate_btn.click(
                    webui.generate_image,
                    inputs=[prompt, negative_prompt, steps, cfg_scale, seed, width, height],
                    outputs=[generated_image, generation_status]
                )
        
        gr.Markdown("---")
        gr.Markdown("💡 **提示：** 首先下载数据集，然后训练你的 LoRA，最后用于推理生成！")
    
    return demo

if __name__ == "__main__":
    # Add gradio to requirements if not present
    try:
        import gradio
    except ImportError:
        print("正在安装 gradio...")
        subprocess.run(["pip", "install", "gradio"], check=True)
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )
