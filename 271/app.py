#!/usr/bin/env python3
"""
Gradio WebUI for HY-WorldPlay Video Generation
支持两种模式：
1. 交互模式 - 使用WASD按键实时控制相机
2. 轨迹模式 - 使用预定义的相机轨迹JSON生成视频

此文件是完全独立的，不需要修改原始代码库中的任何文件。
通过 monkey patching 在运行时修复 pipeline 中的问题。
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gradio as gr
import torch
import json
import numpy as np
import imageio
import einops
from pathlib import Path
import tempfile
from datetime import datetime
from collections import deque
from PIL import Image


# ============== Monkey Patching 修复 Pipeline ==============
# 在导入 pipeline 之前，先准备好补丁

def apply_pipeline_patches():
    """
    应用补丁修复 worldplay_video_pipeline.py 中的问题：
    1. action tensor 必须是 1D（.reshape(-1)），否则 action_in 模块会报错
    2. 确保 viewmats, Ks, action 在正确的设备上
    """
    from hyvideo.pipelines import worldplay_video_pipeline
    from hyvideo.commons import auto_offload_model
    from hyvideo.utils.retrieval_context import select_aligned_memory_frames
    
    def patched_ar_rollout(self, latents, timesteps, prompt_embeds, prompt_mask, 
                           vision_states, cond_latents, task_type, extra_kwargs,
                           viewmats, Ks, action, device):
        """
        修复后的 ar_rollout 方法
        主要修复: action tensor 必须是 1D (.reshape(-1))
        """
        self.init_kv_cache()
        positive_idx = 1 if self.do_classifier_free_guidance else 0
        stabilization_level = 15
        
        # 确保数据在正确设备上
        if viewmats is not None:
            viewmats = viewmats.to(device)
        if Ks is not None:
            Ks = Ks.to(device)
        if action is not None:
            action = action.to(device)
        
        # text, siglip, byt5 embedding cache
        with (torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled),
              auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading)):
            extra_kwargs_pos = {
                "byt5_text_states": extra_kwargs["byt5_text_states"][positive_idx, None, ...],
                "byt5_text_mask": extra_kwargs["byt5_text_mask"][positive_idx, None, ...],
            }
            t_expand_txt = torch.tensor([0]).to(device).to(latents.dtype)
            self._kv_cache = self.transformer(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=t_expand_txt,
                text_states=prompt_embeds[positive_idx, None, ...],
                encoder_attention_mask=prompt_mask[positive_idx, None, ...],
                vision_states=vision_states[positive_idx, None, ...],
                mask_type=task_type,
                extra_kwargs=extra_kwargs_pos,
                kv_cache=self._kv_cache,
                cache_txt=True,
            )
            if self.do_classifier_free_guidance:
                extra_kwargs_neg = {
                    "byt5_text_states": extra_kwargs["byt5_text_states"][0, None, ...],
                    "byt5_text_mask": extra_kwargs["byt5_text_mask"][0, None, ...],
                }
                t_expand_txt = torch.tensor([0]).to(device).to(latents.dtype)
                self._kv_cache_neg = self.transformer(
                    bi_inference=False,
                    ar_txt_inference=True,
                    ar_vision_inference=False,
                    timestep_txt=t_expand_txt,
                    text_states=prompt_embeds[0, None, ...],
                    encoder_attention_mask=prompt_mask[0, None, ...],
                    vision_states=vision_states[0, None, ...],
                    mask_type=task_type,
                    extra_kwargs=extra_kwargs_neg,
                    kv_cache=self._kv_cache_neg,
                    cache_txt=True,
                )

        selected_frame_indices = []

        for chunk_i in range(self.chunk_num):
            if chunk_i > 0:
                current_frame_idx = chunk_i * self.chunk_latent_frames

                selected_frame_indices = []
                for chunk_start_idx in range(current_frame_idx, current_frame_idx + self.chunk_latent_frames, 4):
                    selected_history_frame_id = select_aligned_memory_frames(
                        viewmats[0].cpu().detach().numpy(),
                        chunk_start_idx,
                        memory_frames=20,
                        temporal_context_size=12,
                        pred_latent_size=4,
                        points_local=self.points_local,
                        device=device)
                    selected_frame_indices += selected_history_frame_id
                selected_frame_indices = sorted(list(set(selected_frame_indices)))
                to_remove = list(range(current_frame_idx, current_frame_idx + self.chunk_latent_frames))
                selected_frame_indices = [x for x in selected_frame_indices if x not in to_remove]

                context_latents = latents[:, :, selected_frame_indices]
                context_cond_latents_input = cond_latents[:, :, selected_frame_indices]
                context_latents_input = torch.concat([context_latents, context_cond_latents_input], dim=1)

                context_viewmats = viewmats[:, selected_frame_indices].to(device)
                context_Ks = Ks[:, selected_frame_indices].to(device)
                # 关键修复: action 必须是 1D tensor
                context_action = action[:, selected_frame_indices].reshape(-1).to(device)

                context_timestep = torch.full((len(selected_frame_indices),), stabilization_level - 1,
                                              device=device, dtype=timesteps.dtype)
                # compute kv cache
                with (torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled),
                      auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading)):
                    self._kv_cache = self.transformer(
                        bi_inference=False,
                        ar_txt_inference=False,
                        ar_vision_inference=True,
                        hidden_states=context_latents_input,
                        timestep=context_timestep,
                        timestep_r=None,
                        mask_type=task_type,
                        return_dict=False,
                        viewmats=context_viewmats.to(self.target_dtype),
                        Ks=context_Ks.to(self.target_dtype),
                        action=context_action.to(self.target_dtype),
                        kv_cache=self._kv_cache,
                        cache_vision=True,
                        rope_temporal_size=context_latents_input.shape[2],
                        start_rope_start_idx=0,
                    )
                    if self.do_classifier_free_guidance:
                        self._kv_cache_neg = self.transformer(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            hidden_states=context_latents_input,
                            timestep=context_timestep,
                            timestep_r=None,
                            mask_type=task_type,
                            return_dict=False,
                            viewmats=context_viewmats.to(self.target_dtype),
                            Ks=context_Ks.to(self.target_dtype),
                            action=context_action.to(self.target_dtype),
                            kv_cache=self._kv_cache_neg,
                            cache_vision=True,
                            rope_temporal_size=context_latents_input.shape[2],
                            start_rope_start_idx=0,
                        )

                self.scheduler.set_timesteps(self.num_inference_steps, device=device)

            start_idx = chunk_i * self.chunk_latent_frames
            end_idx = chunk_i * self.chunk_latent_frames + self.chunk_latent_frames

            with self.progress_bar(total=self.num_inference_steps) as progress_bar, \
                 auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading):
                for i, t in enumerate(timesteps):
                    timestep_input = torch.full((self.chunk_latent_frames,), t, device=device,
                                                dtype=timesteps.dtype)
                    latent_model_input = latents[:, :, start_idx: end_idx]
                    cond_latents_input = cond_latents[:, :, start_idx: end_idx]

                    viewmats_input = viewmats[:, start_idx: end_idx].to(device)
                    Ks_input = Ks[:, start_idx: end_idx].to(device)
                    # 关键修复: action 必须是 1D tensor
                    action_input = action[:, start_idx: end_idx].reshape(-1).to(device)

                    latents_concat = torch.concat([latent_model_input, cond_latents_input], dim=1)
                    latents_concat = self.scheduler.scale_model_input(latents_concat, t)

                    with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                        noise_pred = self.transformer(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            hidden_states=latents_concat,
                            timestep=timestep_input,
                            timestep_r=None,
                            mask_type=task_type,
                            return_dict=False,
                            viewmats=viewmats_input.to(self.target_dtype),
                            Ks=Ks_input.to(self.target_dtype),
                            action=action_input.to(self.target_dtype),
                            kv_cache=self._kv_cache,
                            cache_vision=False,
                            rope_temporal_size=latents_concat.shape[2] + len(selected_frame_indices),
                            start_rope_start_idx=len(selected_frame_indices),
                        )[0]
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond = self.transformer(
                                bi_inference=False,
                                ar_txt_inference=False,
                                ar_vision_inference=True,
                                hidden_states=latents_concat,
                                timestep=timestep_input,
                                timestep_r=None,
                                mask_type=task_type,
                                return_dict=False,
                                viewmats=viewmats_input.to(self.target_dtype),
                                Ks=Ks_input.to(self.target_dtype),
                                action=action_input.to(self.target_dtype),
                                kv_cache=self._kv_cache_neg,
                                cache_vision=False,
                                rope_temporal_size=latents_concat.shape[2] + len(selected_frame_indices),
                                start_rope_start_idx=len(selected_frame_indices),
                            )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)

                    latent_model_input = self.scheduler.step(noise_pred, t, latent_model_input, return_dict=False)[0]
                    latents[:, :, start_idx: end_idx] = latent_model_input[:, :, -self.chunk_latent_frames:]

                    if i == len(timesteps) - 1 or ((i + 1) > self.num_warmup_steps
                                                   and (i + 1) % self.scheduler.order == 0):
                        if progress_bar is not None:
                            progress_bar.update()
        
        return latents
    
    def patched_bi_rollout(self, latents, timesteps, prompt_embeds, prompt_mask, 
                           vision_states, cond_latents, task_type, extra_kwargs,
                           viewmats, Ks, action, device):
        """
        修复后的 bi_rollout 方法
        主要修复: 确保数据在正确设备上
        """
        from hyvideo.commons import auto_offload_model
        from hyvideo.utils.retrieval_context import select_aligned_memory_frames
        from hyvideo.pipelines.pipeline_utils import rescale_noise_cfg
        from einops import repeat
        
        # 确保数据在正确设备上
        if viewmats is not None:
            viewmats = viewmats.to(device)
        if Ks is not None:
            Ks = Ks.to(device)
        if action is not None:
            action = action.to(device)
        
        stabilization_level = 15
        for chunk_i in range(self.chunk_num):
            if chunk_i > 0:
                current_frame_idx = chunk_i * self.chunk_latent_frames

                selected_frame_indices = []
                for chunk_start_idx in range(current_frame_idx, current_frame_idx + self.chunk_latent_frames, 4):
                    selected_history_frame_id = select_aligned_memory_frames(
                        viewmats[0].cpu().detach().numpy(),
                        chunk_start_idx,
                        memory_frames=20,
                        temporal_context_size=12,
                        pred_latent_size=4,
                        points_local=self.points_local,
                        device=device)
                    selected_frame_indices = selected_frame_indices + selected_history_frame_id
                selected_frame_indices = sorted(list(set(selected_frame_indices)))
                to_remove = list(range(current_frame_idx, current_frame_idx + self.chunk_latent_frames))
                selected_frame_indices = [x for x in selected_frame_indices if x not in to_remove]

                context_latents = latents[:, :, selected_frame_indices]
                context_w2c = viewmats[:, selected_frame_indices]
                context_Ks = Ks[:, selected_frame_indices]
                context_action = action[:, selected_frame_indices]

                self.scheduler.set_timesteps(self.num_inference_steps, device=device)

            start_idx = chunk_i * self.chunk_latent_frames
            end_idx = chunk_i * self.chunk_latent_frames + self.chunk_latent_frames

            with (self.progress_bar(total=self.num_inference_steps) as progress_bar,
                  auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading)):
                for i, t in enumerate(timesteps):
                    if chunk_i == 0:
                        timestep_input = torch.full((self.chunk_latent_frames,), t,
                                                    device=device, dtype=timesteps.dtype)
                        latent_model_input = latents[:, :, :self.chunk_latent_frames]
                        cond_latents_input = cond_latents[:, :, :self.chunk_latent_frames]
                    else:
                        t_now = torch.full((self.chunk_latent_frames,), t,
                                           device=device, dtype=timesteps.dtype)
                        t_ctx = torch.full((len(selected_frame_indices),), stabilization_level - 1,
                                           device=device, dtype=timesteps.dtype)
                        timestep_input = torch.cat([t_ctx, t_now], dim=0)

                        latents_model_now = latents[:, :, start_idx: end_idx]
                        latent_model_input = torch.cat([context_latents, latents_model_now], dim=2)
                        cond_latents_input = cond_latents[:, :, :latent_model_input.shape[2]]

                    viewmats_input = viewmats[:, start_idx: end_idx]
                    Ks_input = Ks[:, start_idx: end_idx]
                    action_input = action[:, start_idx: end_idx]

                    if chunk_i > 0:
                        viewmats_input = torch.cat([context_w2c, viewmats_input], dim=1)
                        Ks_input = torch.cat([context_Ks, Ks_input], dim=1)
                        action_input = torch.cat([context_action, action_input], dim=1)

                    latents_concat = torch.concat([latent_model_input, cond_latents_input], dim=1)
                    if self.do_classifier_free_guidance:
                        latents_concat = torch.cat([latents_concat] * 2)
                    latents_concat = self.scheduler.scale_model_input(latents_concat, t)

                    batch_size = latents_concat.shape[0]
                    t_expand_txt = t.repeat(batch_size)
                    t_expand = timestep_input.repeat(batch_size)
                    viewmats_input = repeat(viewmats_input, 'B L H W -> (B R) L H W', R=batch_size).to(device)
                    Ks_input = repeat(Ks_input, 'B L H W -> (B R) L H W', R=batch_size).to(device)
                    # 关键: 使用 repeat 后 reshape 成 1D
                    action_input = repeat(action_input, 'B L -> (B R) L', R=batch_size).reshape(-1).to(device)

                    with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                        output = self.transformer(
                            bi_inference=True,
                            ar_txt_inference=False,
                            ar_vision_inference=False,
                            hidden_states=latents_concat,
                            timestep=t_expand,
                            timestep_txt=t_expand_txt,
                            text_states=prompt_embeds,
                            text_states_2=None,
                            encoder_attention_mask=prompt_mask,
                            timestep_r=None,
                            vision_states=vision_states,
                            mask_type=task_type,
                            guidance=None,
                            return_dict=False,
                            extra_kwargs=extra_kwargs,
                            viewmats=viewmats_input.to(self.target_dtype),
                            Ks=Ks_input.to(self.target_dtype),
                            action=action_input.to(self.target_dtype),
                        )
                        noise_pred = output[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    latent_model_input = self.scheduler.step(noise_pred, t, latent_model_input, return_dict=False)[0]
                    latents[:, :, start_idx: end_idx] = latent_model_input[:, :, -self.chunk_latent_frames:]

                    if i == len(timesteps) - 1 or ((i + 1) > self.num_warmup_steps
                                                   and (i + 1) % self.scheduler.order == 0):
                        if progress_bar is not None:
                            progress_bar.update()

        return latents
    
    # 应用补丁
    worldplay_video_pipeline.HunyuanVideo_1_5_Pipeline.ar_rollout = patched_ar_rollout
    worldplay_video_pipeline.HunyuanVideo_1_5_Pipeline.bi_rollout = patched_bi_rollout
    
    print("✅ Pipeline patches applied successfully!")

# 应用补丁
apply_pipeline_patches()

# 现在安全地导入
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from scipy.spatial.transform import Rotation as R


# Global pipeline cache
pipeline_cache = {}
current_config = {}

# Default paths - update these based on your setup
DEFAULT_MODEL_PATH = "./checkpoints/HunyuanVideo-1.5"
DEFAULT_BI_ACTION_PATH = "./checkpoints/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors"
DEFAULT_AR_ACTION_PATH = "./checkpoints/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors"
DEFAULT_AR_DISTILL_ACTION_PATH = "./checkpoints/HY-WorldPlay/ar_distilled_action_model/model.safetensors"


# ============== 交互模式状态管理 ==============
class InteractiveState:
    """管理交互模式的全局状态"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.is_active = False
        self.image_path = None
        self.prompt = ""
        self.action_queue = []  # 动作队列
        self.current_c2w = np.eye(4)  # 当前相机位姿 (camera to world)
        self.video_chunks = []  # 生成的视频片段路径
        self.frame_count = 0
        self.all_videos = []  # 所有生成的视频tensor

interactive_state = InteractiveState()


# ============== 工具函数 ==============
def initialize_env():
    """Initialize parallel state for single GPU"""
    if 'parallel_initialized' not in globals():
        parallel_dims = initialize_parallel_state(sp=1)
        torch.cuda.set_device(0)
        globals()['parallel_initialized'] = True


def pose_to_input(pose_json_path, latent_chunk_num, tps=False):
    """Convert pose JSON to model input format"""
    mapping = {
        (0,0,0,0): 0, (1,0,0,0): 1, (0,1,0,0): 2, (0,0,1,0): 3,
        (0,0,0,1): 4, (1,0,1,0): 5, (1,0,0,1): 6, (0,1,1,0): 7,
        (0,1,0,1): 8,
    }
    
    def one_hot_to_one_dimension(one_hot):
        y = torch.tensor([mapping[tuple(row.tolist())] for row in one_hot])
        return y
    
    pose_json = json.load(open(pose_json_path, 'r'))
    pose_keys = list(pose_json.keys())
    intrinsic_list = []
    w2c_list = []
    
    for i in range(latent_chunk_num):
        t_key = pose_keys[i]
        c2w = np.array(pose_json[t_key]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        intrinsic = np.array(pose_json[t_key]["K"])
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    c2ws = np.linalg.inv(w2c_list)
    C_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0, ...] = c2ws[0, ...]
    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    for i in range(1, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]
        move_norms = np.linalg.norm(move_dirs)
        if move_norms > move_norm_valid:
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
            trans_angles_deg = angles_rad * (180.0 / torch.pi)
        else:
            trans_angles_deg = torch.zeros(3)

        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler('xyz', degrees=True)

        if move_norms > move_norm_valid:
            if (not tps) or (tps == True and abs(rot_angles_deg[1]) < 5e-2 and abs(rot_angles_deg[0]) < 5e-2):
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1
                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1
        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1
            
    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)
    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    action_one_label = trans_one_label * 9 + rotate_one_label

    return torch.tensor(w2c_list), torch.tensor(intrinsic_list), action_one_label


def save_video(video, path):
    """Save video tensor to file"""
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=24)
    return path


def load_pipeline(model_path, action_ckpt, model_type, dtype, enable_sr, enable_offloading=False):
    """Load or retrieve cached pipeline"""
    cache_key = f"{model_path}_{action_ckpt}_{model_type}_{dtype}_{enable_sr}_{enable_offloading}"
    
    if cache_key in pipeline_cache:
        return pipeline_cache[cache_key]
    
    # Clear old cache to save memory
    pipeline_cache.clear()
    
    transformer_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
    
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=model_path,
        transformer_version="480p_i2v",
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_offloading,
        create_sr_pipeline=enable_sr,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=action_ckpt,
    )
    
    pipeline_cache[cache_key] = pipe
    return pipe


# ============== 交互模式函数 ==============

# 动作定义
# trans_one_hot: [前进, 后退, 左移, 右移]
# rotate_one_hot: [左转, 右转, 上看, 下看]
# action = trans * 9 + rotate

def get_action_label(trans_idx, rot_idx):
    """
    计算动作标签
    trans_idx: 0=无, 1=前进, 2=后退, 3=左移, 4=右移
    rot_idx: 0=无, 1=左转, 2=右转, 3=上看, 4=下看
    """
    # 映射到one-hot索引
    trans_mapping = {
        0: 0,  # 无移动 -> (0,0,0,0) -> 0
        1: 1,  # 前进 -> (1,0,0,0) -> 1
        2: 2,  # 后退 -> (0,1,0,0) -> 2
        3: 3,  # 左移 -> (0,0,1,0) -> 3
        4: 4,  # 右移 -> (0,0,0,1) -> 4
    }
    rot_mapping = {
        0: 0,  # 无旋转 -> (0,0,0,0) -> 0
        1: 1,  # 左转 -> (1,0,0,0) -> 1
        2: 2,  # 右转 -> (0,1,0,0) -> 2
        3: 3,  # 上看 -> (0,0,1,0) -> 3
        4: 4,  # 下看 -> (0,0,0,1) -> 4
    }
    return trans_mapping[trans_idx] * 9 + rot_mapping[rot_idx]


def action_to_transform(action_name, move_dist=0.1, rot_deg=5.0):
    """
    将动作名称转换为相机变换矩阵
    返回相对变换 (relative c2w transform)
    """
    transform = np.eye(4)
    
    if action_name == 'W':  # 前进 (沿+Z方向)
        transform[2, 3] = move_dist
    elif action_name == 'S':  # 后退 (沿-Z方向)
        transform[2, 3] = -move_dist
    elif action_name == 'A':  # 左移
        transform[0, 3] = -move_dist
    elif action_name == 'D':  # 右移
        transform[0, 3] = move_dist
    elif action_name == 'LEFT':  # 左转
        r = R.from_euler('y', rot_deg, degrees=True)
        transform[:3, :3] = r.as_matrix()
    elif action_name == 'RIGHT':  # 右转
        r = R.from_euler('y', -rot_deg, degrees=True)
        transform[:3, :3] = r.as_matrix()
    elif action_name == 'UP':  # 上看
        r = R.from_euler('x', rot_deg, degrees=True)
        transform[:3, :3] = r.as_matrix()
    elif action_name == 'DOWN':  # 下看
        r = R.from_euler('x', -rot_deg, degrees=True)
        transform[:3, :3] = r.as_matrix()
    
    return transform


def action_name_to_label(action_name):
    """将动作名称转换为模型需要的action label"""
    action_map = {
        'NONE': get_action_label(0, 0),  # 无动作
        'W': get_action_label(1, 0),     # 前进
        'S': get_action_label(2, 0),     # 后退
        'A': get_action_label(3, 0),     # 左移
        'D': get_action_label(4, 0),     # 右移
        'LEFT': get_action_label(0, 1),  # 左转
        'RIGHT': get_action_label(0, 2), # 右转
        'UP': get_action_label(0, 3),    # 上看
        'DOWN': get_action_label(0, 4),  # 下看
        # 组合动作
        'W+LEFT': get_action_label(1, 1),
        'W+RIGHT': get_action_label(1, 2),
        'S+LEFT': get_action_label(2, 1),
        'S+RIGHT': get_action_label(2, 2),
    }
    return action_map.get(action_name, 0)


def create_trajectory_from_actions(actions, start_c2w=None, latent_chunk_num=None):
    """
    从动作序列创建相机轨迹
    返回: w2c_list, K_list, action_labels
    
    latent_chunk_num: 模型需要的 latent frame 数量
    """
    if start_c2w is None:
        start_c2w = np.eye(4)
    
    c2w_list = [start_c2w.copy()]
    action_labels = [action_name_to_label('NONE')]  # 第一帧无动作
    
    current_c2w = start_c2w.copy()
    
    for action_name in actions:
        # 计算相对变换
        rel_transform = action_to_transform(action_name)
        # 更新位姿: new_c2w = current_c2w @ rel_transform
        current_c2w = current_c2w @ rel_transform
        c2w_list.append(current_c2w.copy())
        action_labels.append(action_name_to_label(action_name))
    
    # 如果指定了 latent_chunk_num，确保长度匹配
    if latent_chunk_num is not None:
        # 获取最后一个动作用于填充（如果没有动作则用NONE）
        last_action = actions[-1] if len(actions) > 0 else 'NONE'
        last_action_label = action_name_to_label(last_action)
        
        while len(c2w_list) < latent_chunk_num:
            # 继续用最后一个动作填充，保持运动连贯性
            rel_transform = action_to_transform(last_action)
            current_c2w = current_c2w @ rel_transform
            c2w_list.append(current_c2w.copy())
            action_labels.append(last_action_label)
        # 截断到正确长度
        c2w_list = c2w_list[:latent_chunk_num]
        action_labels = action_labels[:latent_chunk_num]
    
    # 转换为w2c
    c2w_array = np.array(c2w_list)
    w2c_array = np.linalg.inv(c2w_array)
    
    # 创建内参矩阵 (归一化)
    K = np.array([
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
        [0, 0, 1]
    ])
    K_list = np.array([K] * len(c2w_list))
    
    return (
        torch.tensor(w2c_array, dtype=torch.float32),
        torch.tensor(K_list, dtype=torch.float32),
        torch.tensor(action_labels, dtype=torch.long)
    )


# ============== 轨迹模式生成函数 ==============
def generate_video_gradio(
    image,
    prompt,
    pose_json_path,
    model_type,
    seed,
    video_length,
    num_inference_steps,
    enable_sr,
    negative_prompt,
    aspect_ratio,
    model_path,
    action_ckpt,
    dtype,
    few_step,
    enable_offloading,
    progress=gr.Progress()
):
    """Generate video using Gradio interface"""
    try:
        initialize_env()
        
        # Create a simple InferState-like object
        class SimpleArgs:
            def __init__(self, offload):
                self.offloading = offload
                self.group_offloading = offload
                self.enable_torch_compile = False
                
        initialize_infer_state(SimpleArgs(enable_offloading))
        
        progress(0.1, desc="加载模型中...")
        
        # Load pipeline
        pipe = load_pipeline(model_path, action_ckpt, model_type, dtype, enable_sr, enable_offloading)
        
        progress(0.3, desc="Processing pose data...")
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image.save(tmp_img.name)
            image_path = tmp_img.name
        
        # Load pose data
        latent_chunk_num = (video_length - 1) // 4 + 1
        viewmats, Ks, action = pose_to_input(pose_json_path, latent_chunk_num)
        
        progress(0.5, desc="生成视频中...")
        
        # Generate video
        out = pipe(
            enable_sr=enable_sr,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            sr_num_inference_steps=None,
            video_length=video_length,
            negative_prompt=negative_prompt,
            seed=seed,
            output_type="pt",
            prompt_rewrite=False,  # No vLLM
            return_pre_sr_video=True,
            viewmats=viewmats.unsqueeze(0),
            Ks=Ks.unsqueeze(0),
            action=action.unsqueeze(0),
            few_step=few_step,
            chunk_latent_frames=4 if model_type == "ar" else 16,
            model_type=model_type,
            reference_image=image_path,
        )
        
        progress(0.9, desc="保存视频中...")
        
        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./outputs") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = str(output_dir / "generated.mp4")
        sr_video_path = str(output_dir / "generated_sr.mp4")
        
        if enable_sr and hasattr(out, 'sr_videos'):
            save_video(out.sr_videos, sr_video_path)
            save_video(out.videos, video_path)
            result_video = sr_video_path
            info = f"✅ 视频生成成功！\n\n📁 超分视频: {sr_video_path}\n📁 原始视频: {video_path}"
        else:
            save_video(out.videos, video_path)
            result_video = video_path
            info = f"✅ 视频生成成功！\n\n📁 输出: {video_path}"
        
        # Cleanup
        os.unlink(image_path)
        
        progress(1.0, desc="完成！")
        return result_video, info
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


# ============== 交互模式函数 ==============
def interactive_init(image, prompt, model_path, action_ckpt, dtype):
    """初始化交互会话"""
    global interactive_state
    
    if image is None:
        return "❌ 请先上传起始图片！", "", None
    
    try:
        initialize_env()
        
        class SimpleArgs:
            def __init__(self, offload):
                self.offloading = offload
                self.group_offloading = offload
                self.enable_torch_compile = False
        
        initialize_infer_state(SimpleArgs(False))  # 不使用卸载
        
        # 预加载模型
        pipe = load_pipeline(model_path, action_ckpt, "ar", dtype, False, False)  # 不使用卸载
        
        # 保存图片
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image.save(tmp_img.name)
            image_path = tmp_img.name
        
        # 初始化状态
        interactive_state.reset()
        interactive_state.is_active = True
        interactive_state.image_path = image_path
        interactive_state.prompt = prompt
        interactive_state.current_c2w = np.eye(4)
        
        info = """✅ 会话已启动！

🎮 **操作指南**:
1. 先选择「视频帧数」(设置中)
2. 查看需要多少个动作
3. 点击方向按钮添加动作
4. 点击「生成视频」执行

⚡ W/S: 前进/后退 | A/D: 左移/右移
⚡ ←/→: 左转/右转 | ↑/↓: 上看/下看"""
        
        # 默认29帧需要7个动作
        return info, get_queue_display(29), image
        
    except Exception as e:
        import traceback
        return f"❌ 初始化失败: {str(e)}\n{traceback.format_exc()}", "", None


def interactive_add_action(action_name, video_frames=29):
    """添加动作到队列"""
    global interactive_state
    
    if not interactive_state.is_active:
        return "⚠️ 请先点击「启动会话」", get_queue_display(video_frames)
    
    interactive_state.action_queue.append(action_name)
    
    action_display = {
        'W': '↑前进', 'S': '↓后退', 'A': '←左移', 'D': '→右移',
        'LEFT': '↰左转', 'RIGHT': '↱右转', 'UP': '↑上看', 'DOWN': '↓下看'
    }
    
    status = f"✅ 已添加: {action_display.get(action_name, action_name)}"
    
    return status, get_queue_display(video_frames)


def get_queue_display(video_frames=29):
    """获取队列显示字符串，包含动作计数"""
    action_display = {
        'W': '↑前进', 'S': '↓后退', 'A': '←左移', 'D': '→右移',
        'LEFT': '↰左转', 'RIGHT': '↱右转', 'UP': '↑上看', 'DOWN': '↓下看'
    }
    
    # 计算需要的动作数量
    latent_chunk_num = (video_frames - 1) // 4 + 1
    required_actions = latent_chunk_num - 1  # 第一帧是静止的
    current_count = len(interactive_state.action_queue)
    
    if not interactive_state.action_queue:
        return f"📋 动作队列: [空]\n📊 已选: 0/{required_actions} 个动作"
    
    queue_str = " → ".join([action_display.get(a, a) for a in interactive_state.action_queue])
    
    if current_count < required_actions:
        hint = f"⚠️ 还需 {required_actions - current_count} 个动作 (不足部分将重复最后动作)"
    elif current_count == required_actions:
        hint = "✅ 动作数量刚好!"
    else:
        hint = f"⚠️ 超出 {current_count - required_actions} 个动作 (将被忽略)"
    
    return f"📋 动作队列: [{queue_str}]\n📊 已选: {current_count}/{required_actions} 个动作\n{hint}"


def interactive_clear_queue(video_frames=29):
    """清空动作队列"""
    global interactive_state
    interactive_state.action_queue = []
    return "✅ 队列已清空", get_queue_display(video_frames)


def interactive_generate(model_path, action_ckpt, dtype, num_steps, video_frames, progress=gr.Progress()):
    """根据队列中的动作生成视频"""
    global interactive_state
    
    video_frames = int(video_frames)
    
    if not interactive_state.is_active:
        return "⚠️ 请先点击「启动会话」初始化", get_queue_display(video_frames), None
    
    if len(interactive_state.action_queue) == 0:
        return "⚠️ 动作队列为空，请先添加动作", get_queue_display(video_frames), None
    
    try:
        progress(0.1, desc="准备生成...")
        
        # 获取动作
        actions = interactive_state.action_queue.copy()
        
        # 计算视频长度和 latent chunk 数量
        # video_length = latent_chunk_num * 4 + 1 (每个latent chunk对应4个pixel frames)
        video_length = int(video_frames)  # 用户设置的帧数
        latent_chunk_num = (video_length - 1) // 4 + 1
        
        # 根据帧数计算需要的动作数量
        max_actions = latent_chunk_num - 1  # 第一帧是NONE，其余每个chunk一个动作
        current_actions = actions[:max_actions]
        
        # 创建轨迹，确保长度匹配
        w2c_list, K_list, action_labels = create_trajectory_from_actions(
            current_actions, 
            interactive_state.current_c2w,
            latent_chunk_num=latent_chunk_num
        )
        
        progress(0.3, desc="加载模型...")
        
        # 重新初始化环境确保状态正确
        class SimpleArgs:
            def __init__(self, offload):
                self.offloading = offload
                self.group_offloading = offload
                self.enable_torch_compile = False
        
        initialize_infer_state(SimpleArgs(False))  # 不使用卸载
        
        # 加载pipeline
        pipe = load_pipeline(model_path, action_ckpt, "ar", dtype, False, False)  # 不使用卸载
        
        progress(0.5, desc="生成视频中...")
        
        # 生成
        out = pipe(
            enable_sr=False,
            prompt=interactive_state.prompt,
            aspect_ratio="16:9",
            num_inference_steps=num_steps,
            video_length=video_length,
            seed=None,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=True,
            viewmats=w2c_list.unsqueeze(0),
            Ks=K_list.unsqueeze(0),
            action=action_labels.unsqueeze(0),
            few_step=False,  # 完整模型不用 few_step
            chunk_latent_frames=4,
            model_type='ar',
            reference_image=interactive_state.image_path,
        )
        
        progress(0.9, desc="保存视频...")
        
        # 保存视频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = Path("./outputs/interactive") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = str(output_dir / "chunk.mp4")
        save_video(out.videos, video_path)
        
        # 更新状态
        # 更新相机位姿到最后一帧
        for action_name in current_actions:
            rel_transform = action_to_transform(action_name)
            interactive_state.current_c2w = interactive_state.current_c2w @ rel_transform
        
        # 从队列中移除已执行的动作
        interactive_state.action_queue = actions[max_actions:]
        
        # 记录
        interactive_state.video_chunks.append(video_path)
        interactive_state.frame_count += video_length - 1
        
        progress(1.0, desc="完成!")
        
        used_actions = len(current_actions)
        remaining_actions = len(interactive_state.action_queue)
        
        status = f"""✅ 生成成功！
📽️ 本次生成: {video_length}帧 (约{video_length/24:.1f}秒)
🎮 使用动作: {used_actions}个，剩余: {remaining_actions}个
🎬 累计帧数: {interactive_state.frame_count}
📁 保存至: {video_path}"""
        
        return status, get_queue_display(video_frames), video_path
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        return error_msg, get_queue_display(video_frames), None


def interactive_stop():
    """停止会话"""
    global interactive_state
    
    if not interactive_state.is_active:
        return "⚠️ 没有活动的会话", None
    
    # 清理
    if interactive_state.image_path and os.path.exists(interactive_state.image_path):
        try:
            os.unlink(interactive_state.image_path)
        except:
            pass
    
    total_frames = interactive_state.frame_count
    chunks = len(interactive_state.video_chunks)
    last_video = interactive_state.video_chunks[-1] if interactive_state.video_chunks else None
    
    interactive_state.reset()
    
    info = f"""✅ 会话已结束

📊 统计:
- 总帧数: {total_frames}
- 视频片段: {chunks}个

💾 视频已保存到 ./outputs/interactive/ 目录"""
    
    return info, last_video


# ============== UI 构建 ==============
def create_ui():
    """Create Gradio UI with tabs"""
    
    with gr.Blocks(title="HY-WorldPlay 视频生成器", theme=gr.themes.Soft(), css="""
        .action-btn { min-width: 80px !important; }
        .big-btn { min-height: 50px !important; font-size: 18px !important; }
    """) as demo:
        
        gr.Markdown("""
        # 🎮 HY-WorldPlay 视频生成器
        
        使用 HunyuanVideo-1.5 生成可控相机轨迹视频 | 支持 **交互模式** (WASD控制) 和 **轨迹模式** (JSON定义)
        """)
        
        with gr.Tabs():
            # ============== 交互模式 Tab ==============
            with gr.TabItem("🎮 交互模式 (WASD控制)", id="interactive"):
                gr.Markdown("""
                ### 实时控制相机探索世界
                使用按钮控制相机移动和旋转，生成交互式视频！
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1️⃣ 初始化")
                        
                        inter_image = gr.Image(
                            label="起始图片",
                            type="pil",
                            height=250
                        )
                        
                        inter_prompt = gr.Textbox(
                            label="场景描述",
                            value="A beautiful outdoor scene with natural lighting.",
                            lines=2
                        )
                        
                        with gr.Accordion("⚙️ 模型设置", open=False):
                            inter_model_path = gr.Textbox(
                                label="模型路径",
                                value=DEFAULT_MODEL_PATH
                            )
                            inter_action_ckpt = gr.Textbox(
                                label="动作模型",
                                value=DEFAULT_AR_ACTION_PATH
                            )
                            inter_dtype = gr.Radio(
                                choices=["bf16", "fp32"],
                                value="bf16",
                                label="精度"
                            )
                            inter_steps = gr.Slider(
                                minimum=10, maximum=50, value=50, step=1,
                                label="推理步数 (完整模型建议50步)"
                            )
                            # 合法的帧数: latent_frames 必须能被 4 整除
                            # 13帧(4latent), 29帧(8latent), 45帧(12latent), 61帧(16latent)
                            inter_video_frames = gr.Dropdown(
                                choices=[
                                    ("13帧 (3动作) ≈0.5秒", 13),
                                    ("29帧 (7动作) ≈1.2秒", 29),
                                    ("45帧 (11动作) ≈1.9秒", 45),
                                    ("61帧 (15动作) ≈2.5秒", 61),
                                ],
                                value=29,
                                label="视频帧数",
                                info="必须选择有效帧数，否则最后几帧会花屏"
                            )
                        
                        init_btn = gr.Button("🚀 启动会话", variant="primary", elem_classes="big-btn")
                        
                        gr.Markdown("---")
                        gr.Markdown("#### 2️⃣ 控制面板")
                        
                        # WASD 控制按钮
                        with gr.Group():
                            gr.Markdown("**移动控制**")
                            with gr.Row():
                                gr.Column(scale=1)
                                w_btn = gr.Button("W ↑\n前进", elem_classes="action-btn")
                                gr.Column(scale=1)
                            with gr.Row():
                                a_btn = gr.Button("A ←\n左移", elem_classes="action-btn")
                                s_btn = gr.Button("S ↓\n后退", elem_classes="action-btn")
                                d_btn = gr.Button("D →\n右移", elem_classes="action-btn")
                        
                        with gr.Group():
                            gr.Markdown("**视角控制**")
                            with gr.Row():
                                gr.Column(scale=1)
                                up_btn = gr.Button("↑\n上看", elem_classes="action-btn")
                                gr.Column(scale=1)
                            with gr.Row():
                                left_btn = gr.Button("←\n左转", elem_classes="action-btn")
                                down_btn = gr.Button("↓\n下看", elem_classes="action-btn")
                                right_btn = gr.Button("→\n右转", elem_classes="action-btn")
                        
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ 清空队列")
                            gen_btn = gr.Button("🎬 生成视频", variant="primary", elem_classes="big-btn")
                        
                        stop_btn = gr.Button("⏹️ 结束会话", variant="stop")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 输出")
                        
                        inter_status = gr.Textbox(
                            label="状态",
                            lines=8,
                            interactive=False
                        )
                        
                        inter_queue = gr.Textbox(
                            label="动作队列",
                            value="📋 动作队列: [空]\n📊 已选: 0/7 个动作",
                            lines=3,
                            interactive=False
                        )
                        
                        inter_preview = gr.Image(
                            label="当前画面",
                            height=200
                        )
                        
                        inter_video = gr.Video(
                            label="生成的视频",
                            height=300
                        )
                
                # 绑定事件
                init_btn.click(
                    fn=interactive_init,
                    inputs=[inter_image, inter_prompt, inter_model_path, inter_action_ckpt, inter_dtype],
                    outputs=[inter_status, inter_queue, inter_preview]
                )
                
                # 动作按钮 - 传递视频帧数以显示正确的动作数量提示
                w_btn.click(fn=lambda vf: interactive_add_action('W', vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                s_btn.click(fn=lambda vf: interactive_add_action('S', vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                a_btn.click(fn=lambda vf: interactive_add_action('A', vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                d_btn.click(fn=lambda vf: interactive_add_action('D', vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                left_btn.click(fn=lambda vf: interactive_add_action('LEFT', vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                right_btn.click(fn=lambda vf: interactive_add_action('RIGHT', vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                up_btn.click(fn=lambda vf: interactive_add_action('UP', vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                down_btn.click(fn=lambda vf: interactive_add_action('DOWN', vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                
                clear_btn.click(fn=lambda vf: interactive_clear_queue(vf), inputs=[inter_video_frames], outputs=[inter_status, inter_queue])
                
                # 当视频帧数改变时更新队列显示
                inter_video_frames.change(fn=lambda vf: get_queue_display(vf), inputs=[inter_video_frames], outputs=[inter_queue])
                
                gen_btn.click(
                    fn=interactive_generate,
                    inputs=[inter_model_path, inter_action_ckpt, inter_dtype, inter_steps, inter_video_frames],
                    outputs=[inter_status, inter_queue, inter_video]
                )
                
                stop_btn.click(fn=interactive_stop, outputs=[inter_status, inter_video])
                
                # 交互模式示例
                gr.Examples(
                    examples=[
                        [
                            "./assets/img/test.png",
                            "A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path.",
                        ],
                    ],
                    inputs=[inter_image, inter_prompt],
                    label="📷 示例图片"
                )
            
            # ============== 轨迹模式 Tab ==============
            with gr.TabItem("📐 轨迹模式 (JSON定义)", id="trajectory"):
                gr.Markdown("""
                ### 使用预定义相机轨迹生成视频
                通过 JSON 文件精确控制每一帧的相机位置和朝向
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 输入设置")
                        
                        image_input = gr.Image(
                            label="输入图片（I2V 必需）",
                            type="pil",
                            height=300
                        )
                        
                        prompt_input = gr.Textbox(
                            label="提示词",
                            placeholder="Describe your scene...",
                            lines=4,
                            value="A paved pathway leads towards a stone arch bridge spanning a calm body of water."
                        )
                        
                        negative_prompt_input = gr.Textbox(
                            label="负面提示词（可选）",
                            placeholder="What you don't want in the video...",
                            lines=2,
                            value=""
                        )
                        
                        pose_json_input = gr.Textbox(
                            label="位姿 JSON 路径",
                            value="./assets/pose/test_forward_32_latents.json",
                            info="相机轨迹 JSON 文件路径"
                        )
                        
                        with gr.Row():
                            model_type_input = gr.Radio(
                                choices=["bi", "ar", "ar_distilled"],
                                value="bi",
                                label="模型类型",
                                info="bi=双向模型（质量更高）, ar=自回归模型（更快）, ar_distilled=最快"
                            )
                        
                        with gr.Accordion("高级设置", open=False):
                            seed_input = gr.Slider(
                                minimum=0,
                                maximum=2147483647,
                                value=1,
                                step=1,
                                label="随机种子"
                            )
                            
                            video_length_input = gr.Slider(
                                minimum=33,
                                maximum=125,
                                value=125,
                                step=4,
                                label="视频长度（帧数）",
                                info="必须是 4n+1 格式（如 33, 37, 41, ..., 125）"
                            )
                            
                            num_steps_input = gr.Slider(
                                minimum=4,
                                maximum=50,
                                value=50,
                                step=1,
                                label="推理步数",
                                info="步数越多质量越好，但速度越慢"
                            )
                            
                            aspect_ratio_input = gr.Dropdown(
                                choices=["16:9", "9:16", "4:3", "3:4", "1:1"],
                                value="16:9",
                                label="宽高比"
                            )
                            
                            enable_sr_input = gr.Checkbox(
                                label="启用超分辨率",
                                value=False,
                                info="仅在 video_length=121 时有效"
                            )
                            
                            few_step_input = gr.Checkbox(
                                label="少步模式",
                                value=False,
                                info="仅用于蒸馏模型"
                            )
                            
                            dtype_input = gr.Radio(
                                choices=["bf16", "fp32"],
                                value="bf16",
                                label="精度",
                                info="bf16=更快, fp32=质量更好"
                            )
                            
                            enable_offloading_input = gr.Checkbox(
                                label="启用卸载",
                                value=False,
                                info="关闭以获得更快速度（需要大显存）"
                            )
                        
                        with gr.Accordion("模型路径", open=False):
                            model_path_input = gr.Textbox(
                                label="HunyuanVideo 模型路径",
                                value=DEFAULT_MODEL_PATH
                            )
                            
                            action_ckpt_input = gr.Textbox(
                                label="动作模型路径",
                                value=DEFAULT_BI_ACTION_PATH,
                                info="会根据模型类型自动更新"
                            )
                        
                        generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 输出")
                        
                        video_output = gr.Video(
                            label="生成的视频",
                            height=400
                        )
                        
                        info_output = gr.Textbox(
                            label="状态",
                            lines=6,
                            max_lines=10
                        )
                        
                        gr.Markdown("""
                        ### 📝 使用提示:
                        - **图片**: I2V 生成必需
                        - **位姿 JSON**: 使用自定义 JSON 定义相机轨迹
                        - **模型类型**: 
                          - `bi` (双向): 质量最好，较慢
                          - `ar` (自回归): 平衡选择
                          - `ar_distilled`: 最快，需启用"少步模式"
                        - **视频长度**: 必须是 4n+1（如 33, 37, 41, ..., 125）
                        
                        ### 📂 预设轨迹:
                        - `./assets/pose/test_forward_32_latents.json` - 向前运动
                        """)
                
                # Auto-update action checkpoint path based on model type
                def update_action_path(model_type):
                    if model_type == "bi":
                        return DEFAULT_BI_ACTION_PATH
                    elif model_type == "ar":
                        return DEFAULT_AR_ACTION_PATH
                    elif model_type == "ar_distilled":
                        return DEFAULT_AR_DISTILL_ACTION_PATH
                    return DEFAULT_BI_ACTION_PATH
                
                model_type_input.change(
                    fn=update_action_path,
                    inputs=[model_type_input],
                    outputs=[action_ckpt_input]
                )
                
                # Generate button click
                generate_btn.click(
                    fn=generate_video_gradio,
                    inputs=[
                        image_input,
                        prompt_input,
                        pose_json_input,
                        model_type_input,
                        seed_input,
                        video_length_input,
                        num_steps_input,
                        enable_sr_input,
                        negative_prompt_input,
                        aspect_ratio_input,
                        model_path_input,
                        action_ckpt_input,
                        dtype_input,
                        few_step_input,
                        enable_offloading_input,
                    ],
                    outputs=[video_output, info_output]
                )
                
                # 轨迹模式示例
                gr.Examples(
                    examples=[
                        [
                            "./assets/img/test.png",
                            "A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path.",
                            "./assets/pose/test_forward_32_latents.json",
                            "bi",
                            1,
                            125,
                            50,
                        ],
                    ],
                    inputs=[
                        image_input,
                        prompt_input,
                        pose_json_input,
                        model_type_input,
                        seed_input,
                        video_length_input,
                        num_steps_input,
                    ],
                    label="📷 示例"
                )
        
        gr.Markdown("""
        ---
        ### 📚 关于
        
        **HY-WorldPlay** 是腾讯混元团队开源的实时交互世界模型。
        
        - 🔗 [GitHub](https://github.com/Tencent-Hunyuan/HY-WorldPlay) 
        - 🌐 [官方Demo](https://3d.hunyuan.tencent.com/sceneTo3D) (完整实时交互体验)
        - 📄 [技术报告](https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf)
        
        ⚠️ **注意**: 本地交互模式是简化实现，完整的实时流式交互请访问官方Demo。
        """)
    
    return demo


def preload_models():
    """启动时预加载模型"""
    print("\n" + "="*60)
    print("🚀 正在预加载模型...")
    print("="*60 + "\n")
    
    initialize_env()
    
    class SimpleArgs:
        def __init__(self):
            self.offloading = False
            self.group_offloading = False
            self.enable_torch_compile = False
    
    initialize_infer_state(SimpleArgs())
    
    # 预加载 AR 模型（交互模式使用）
    print("📦 加载 AR 模型 (交互模式)...")
    load_pipeline(
        DEFAULT_MODEL_PATH, 
        DEFAULT_AR_ACTION_PATH, 
        "ar", 
        "bf16", 
        enable_sr=False, 
        enable_offloading=False
    )
    print("✅ AR 模型加载完成!\n")
    
    # 预加载双向模型（轨迹模式使用）
    print("📦 加载双向模型 (轨迹模式)...")
    load_pipeline(
        DEFAULT_MODEL_PATH, 
        DEFAULT_BI_ACTION_PATH, 
        "bi", 
        "bf16", 
        enable_sr=False, 
        enable_offloading=False
    )
    print("✅ 双向模型加载完成!\n")
    
    print("="*60)
    print("🎉 所有模型预加载完成！可以开始使用了。")
    print("="*60 + "\n")


if __name__ == "__main__":
    # 启动 Gradio UI（模型将在首次使用时加载）
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
