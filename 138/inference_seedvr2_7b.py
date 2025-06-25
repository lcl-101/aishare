# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 强制只用单卡
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Monkey-patch distributed/sequence parallel functions for single-GPU
try:
    import types
    import torch.distributed as dist
    def _single_gpu_rank():
        return 0
    def _single_gpu_ranks():
        return [0]
    if not dist.is_available() or not dist.is_initialized():
        import common.distributed.advanced as adv
        adv.get_next_sequence_parallel_rank = _single_gpu_rank
        adv.get_sequence_parallel_global_ranks = _single_gpu_ranks
except Exception as e:
    print(f"[WARN] Could not patch distributed functions: {e}")

import torch
# 移除禁用 cuDNN，恢复默认

import mediapy
from einops import rearrange
from omegaconf import OmegaConf
print(os.getcwd())
import datetime
from tqdm import tqdm
from models.dit import na
import gc

from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
if os.path.exists("./projects/video_diffusion_sr/color_fix.py"):
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    use_colorfix=True
else:
    use_colorfix = False
    print('Note!!!!!! Color fix is not avaliable!')
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io.video import read_video
from torchvision.io import read_image


from common.config import load_config
from common.seed import set_seed
from common.partition import partition_by_groups, partition_by_size
import argparse
from projects.video_diffusion_sr.infer import VideoDiffusionInfer


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def configure_runner(sp_size):
    config_path = os.path.join('./configs_7b', 'main.yaml')
    config = load_config(config_path)
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    runner.configure_dit_model(device="cuda", checkpoint='./ckpts/seedvr2_ema_7b.pth')
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    runner.configure_diffusion()  # 确保 schedule 被初始化
    return runner

def generation_step(runner, text_embeds_dict, cond_latents):
    # 保证所有 embedding 和 latent 都是 float32，避免 dtype 不一致
    for k in text_embeds_dict:
        text_embeds_dict[k] = [emb.float() for emb in text_embeds_dict[k]]
    if isinstance(cond_latents, torch.Tensor):
        cond_latents = cond_latents.float()
    elif isinstance(cond_latents, list):
        cond_latents = [x.float() for x in cond_latents]
    noises = [torch.randn_like(latent) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent) for latent in cond_latents]
    print(f"Generating with noise shape: {noises[0].size()}.")
    # 不做 sync_data，不做多卡分发
    noises = [n.to(get_device()) for n in noises]
    aug_noises = [n.to(get_device()) for n in aug_noises]
    cond_latents = [n.to(get_device()) for n in cond_latents]
    cond_noise_scale = 0.0
    def _add_noise(x, aug_noise):
        t = torch.tensor([1000.0], device=get_device()) * cond_noise_scale
        shape = torch.tensor(x.shape[1:], device=get_device())[None]
        t = runner.timestep_transform(t, shape)
        print(f"Timestep shifting from {1000.0 * cond_noise_scale} to {t}.")
        x = runner.schedule.forward(x, aug_noise, t)
        return x
    conditions = [
        runner.get_condition(
            noise,
            task="sr",
            latent_blur=_add_noise(latent_blur, aug_noise),
        )
        for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
    ]
    with torch.no_grad():
        video_tensors = runner.inference(
            noises=noises,
            conditions=conditions,
            dit_offload=True,
            **text_embeds_dict,
        )
    samples = [
        (
            rearrange(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange(video, "c t h w -> t c h w")
        )
        for video in video_tensors
    ]
    del video_tensors
    return samples

def generation_loop(runner, video_path='./test_videos', output_dir='./results', batch_size=1, cfg_scale=1.0, cfg_rescale=0.0, sample_steps=1, seed=666, res_h=1280, res_w=720, sp_size=1, out_fps=None):
    def _build_pos_and_neg_prompt():
        positive_text = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, \
        hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, \
        skin pore detailing, hyper sharpness, perfect without deformations."
        negative_text = "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, \
        CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, \
        signature, jpeg artifacts, deformed, lowres, over-smooth"
        return positive_text, negative_text
    def _build_test_prompts(video_path):
        positive_text, negative_text = _build_pos_and_neg_prompt()
        original_videos = []
        prompts = {}
        video_list = os.listdir(video_path)
        for f in video_list:
            original_videos.append(f)
            prompts[f] = positive_text
        print(f"Total prompts to be generated: {len(original_videos)}")
        return original_videos, prompts, negative_text
    def _extract_text_embeds():
        positive_prompts_embeds = []
        for texts_pos in original_videos_local:
            text_pos_embeds = torch.load('pos_emb.pt')
            text_neg_embeds = torch.load('neg_emb.pt')
            positive_prompts_embeds.append(
                {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
            )
        gc.collect()
        torch.cuda.empty_cache()
        return positive_prompts_embeds
    # 单卡直接全部推理
    original_videos, _, _ = _build_test_prompts(video_path)
    original_videos_local = [original_videos]  # 不做分组
    positive_prompts_embeds = _extract_text_embeds()
    video_transform = Compose([
        NaResize(resolution=(res_h, res_w), mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])
    for videos, text_embeds in tqdm(zip(original_videos_local, positive_prompts_embeds)):
        cond_latents = []
        fps_lists = []
        for video in videos:
            if is_image_file(video):
                video = read_image(os.path.join(video_path, video)).unsqueeze(0) / 255.0
                if sp_size > 1:
                    raise ValueError("Sp size should be set to 1 for image inputs!")
            else:
                video, _, info = read_video(os.path.join(video_path, video), output_format="TCHW")
                video = video / 255.0
                fps_lists.append(info["video_fps"] if out_fps is None else out_fps)
            print(f"Read video size: {video.size()}")
            transformed = video_transform(video.to(get_device()))
            print(f"After video_transform: {transformed.size()}")
            transformed = pad_h_to_4n1(transformed)
            print(f"After pad_h_to_4n1: {transformed.size()}")
            cond_latents.append(transformed)
        ori_lengths = [video.size(1) for video in cond_latents]
        input_videos = cond_latents
        cond_latents = [cut_videos(video, sp_size) for video in cond_latents]
        for idx, v in enumerate(cond_latents):
            print(f"After cut_videos cond_latents[{idx}]: {v.size()}")
            if v.shape[0] == 3 and v.shape[1] > 3:
                pass
            elif v.shape[1] == 3 and v.shape[0] > 3:
                cond_latents[idx] = v.permute(1, 0, 2, 3)
                print(f"Permuted cond_latents[{idx}] to: {cond_latents[idx].size()}")
            else:
                raise ValueError(f"cond_latents[{idx}] shape not understood: {v.size()}")
        runner.dit.to("cpu")
        print(f"Encoding videos: {list(map(lambda x: x.size(), cond_latents))}")
        runner.vae.to(get_device())
        if isinstance(cond_latents, list):
            cond_latents = torch.stack(cond_latents, dim=0)
        print(f"Input to vae_encode (before permute): {cond_latents.shape}")
        if cond_latents.shape[1] != 3 and cond_latents.shape[2] == 3:
            print(f"[DEBUG] Permuting cond_latents from {cond_latents.shape} to [B, 3, T, H, W]")
            cond_latents = cond_latents.permute(0, 2, 1, 3, 4)
        elif cond_latents.shape[1] != 3:
            raise ValueError(f"cond_latents shape not understood: {cond_latents.shape}")
        print(f"Input to vae_encode: {cond_latents.shape}")
        assert cond_latents.ndim == 5 and cond_latents.shape[1] == 3, f"vae input shape error: {cond_latents.shape}"
        cond_latents = runner.vae_encode(cond_latents)
        if isinstance(cond_latents, list):
            print(f"Output from vae_encode: {[x.shape for x in cond_latents]}")
            try:
                cond_latents = torch.stack(cond_latents, dim=0)
            except Exception as e:
                print(f"[WARN] Cannot stack cond_latents: {e}")
        else:
            print(f"Output from vae_encode: {cond_latents.shape}")
        runner.vae.to("cpu")
        runner.dit.to(get_device())
        for i, emb in enumerate(text_embeds["texts_pos"]):
            text_embeds["texts_pos"][i] = emb.to(get_device())
        for i, emb in enumerate(text_embeds["texts_neg"]):
            text_embeds["texts_neg"][i] = emb.to(get_device())
        samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
        runner.dit.to("cpu")
        del cond_latents
        # dump samples to the output directory
        for path, input, sample, ori_length, save_fps in zip(
            videos, input_videos, samples, ori_lengths, fps_lists
        ):
            if ori_length < sample.shape[0]:
                sample = sample[:ori_length]
            filename = os.path.join(output_dir, os.path.basename(path))
            input = (
                rearrange(input[:, None], "c t h w -> t c h w")
                if input.ndim == 3
                else rearrange(input, "c t h w -> t c h w")
            )
            if use_colorfix:
                sample = wavelet_reconstruction(
                    sample.to("cpu"), input[: sample.size(0)].to("cpu")
                )
            else:
                sample = sample.to("cpu")
            sample = (
                rearrange(sample[:, None], "t c h w -> t h w c")
                if sample.ndim == 3
                else rearrange(sample, "t c h w -> t h w c")
            )
            sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
            sample = sample.to(torch.uint8).numpy()
            if sample.shape[0] == 1:
                mediapy.write_image(filename, sample.squeeze(0))
            else:
                mediapy.write_video(
                    filename, sample, fps=save_fps
                )
        gc.collect()
        torch.cuda.empty_cache()

def is_image_file(filename):
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_exts

def pad_h_to_4n1(x, multiple=16):
    """
    Pads the height of a tensor to a multiple of `multiple` (default 16).
    Supports [C, T, H, W] or [B, C, T, H, W] tensors.
    """
    if x.ndim == 4:
        c, t, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        if pad_h > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_h))
        return x
    elif x.ndim == 5:
        b, c, t, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        if pad_h > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_h))
        return x
    else:
        raise ValueError(f"Unsupported tensor shape for pad_h_to_4n1: {x.shape}")

def cut_videos(x, sp_size=1):
    """
    Pads the T axis (temporal) to 4n+1 as required by VAE.
    Supports [C, T, H, W] or [B, C, T, H, W] tensors.
    """
    if x.ndim == 4:
        c, t, h, w = x.shape
        pad_t = (4 - ((t - 1) % 4)) % 4
        if pad_t > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_t))
        return x
    elif x.ndim == 5:
        b, c, t, h, w = x.shape
        pad_t = (4 - ((t - 1) % 4)) % 4
        if pad_t > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_t))
        return x
    else:
        raise ValueError(f"Unsupported tensor shape for cut_videos: {x.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)
    args = parser.parse_args()

    runner = configure_runner(args.sp_size)
    generation_loop(runner, **vars(args))
