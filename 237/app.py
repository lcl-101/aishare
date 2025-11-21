import argparse
import json
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr 
import numpy as np
import torch
from PIL import Image

import tools.infer_video_720p as infer_720
import tools.infer_video_480p as infer_480
from infinity.utils.arg_util import Args
from infinity.schedules import get_encode_decode_func
from infinity.schedules.dynamic_resolution import (
    get_dynamic_resolution_meta,
    get_first_full_spatial_size_scale_index,
)
from infinity.utils.video_decoder import EncodedVideoOpencv
from tools.run_infinity import (
    gen_one_example,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    transform,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_CHECKPOINT_ROOT = os.environ.get("INFINITYSTAR_CKPT_DIR", "checkpoints/InfinityStar")
SAVE_VIDEO = infer_720.save_video

PIPE_CONFIGS: Dict[str, Dict] = {
    "720p": {
        "tab_label": "720p",
        "weights_subdir": "infinitystar_8b_720p_weights",
        "pn": "0.90M",
        "video_frames": 81,
        "image_scale_repetition": '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]',
        "video_scale_repetition": '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]',
        "detail_scale_min_tokens": 750,
        "semantic_scales": 12,
        "pipe_cls": infer_720.InferencePipe,
        "perform_inference": infer_720.perform_inference,
        "description": "生成约 4 秒（≈5 秒标签）的 720p 文本/图像到视频。",
        "examples": [
            [
                "A handsome smiling gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic",
                41,
                os.path.join("assets", "reference_image.webp"),
                False,
            ],
            [
                "Slow push-in shot of a neon-lit cyberpunk alley during gentle rain, reflective puddles shimmering, cinematic lighting",
                41,
                None,
                False,
            ],
            [
                "Golden-hour aerial view of terraced rice fields with low fog rolling through the valley, hyper-detailed, 4k",
                41,
                None,
                False,
            ],
        ],
    },
    "480p": {
        "tab_label": "480p",
        "weights_subdir": "infinitystar_8b_480p_weights",
        "pn": "0.40M",
        "video_frames": 161,
        "image_scale_repetition": '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]',
        "video_scale_repetition": '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]',
        "detail_scale_min_tokens": 350,
        "semantic_scales": 11,
        "pipe_cls": infer_480.InferencePipe,
        "perform_inference": infer_480.perform_inference,
        "arg_overrides": {"videovae": 10},
        "duration_options": [5, 10],
        "default_duration": 5,
        "description": "灵活的 480p 生成（默认 5 秒，可选 10 秒并支持视频续写）。",
        "examples": [
            [
                "A handsome smiling gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic",
                41,
                os.path.join("assets", "reference_image.webp"),
                False,
                5,
                None,
            ],
            [
                "Handheld shot of a surfer catching a sunset wave, spray sparkling in backlight, dreamy film grain",
                7,
                None,
                False,
                5,
                None,
            ],
            [
                "A cozy campfire scene with friends roasting marshmallows under a starry sky, cinematic lighting",
                99,
                None,
                False,
                10,
                None,
            ],
        ],
    },
}

INTERACTIVE_MODEL_SUBDIR = "InfinityStarInteract_24K_iters"
INTERACTIVE_SEGMENT_SECONDS = 5
INTERACTIVE_REFERENCE_FRAMES = INTERACTIVE_SEGMENT_SECONDS * 16 + 1  # 81 frames
MAX_INTERACTIVE_PROMPTS = 5
INTERACTIVE_TOY_DIR = os.path.join("data", "interactive_toy_videos")


class PipeCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int], Tuple[object, Args]] = {}

    def get(self, config_name: str, enable_rewriter: bool) -> Tuple[object, Args]:
        key = (config_name, int(enable_rewriter))
        if key not in self._cache:
            self._cache[key] = self._build_pipe(config_name, enable_rewriter)
        return self._cache[key]

    def _build_pipe(self, config_name: str, enable_rewriter: bool) -> Tuple[object, Args]:
        config = PIPE_CONFIGS[config_name]
        args = build_args_for_config(config, enable_rewriter)
        torch.cuda.empty_cache()
        pipe = config["pipe_cls"](args)
        return pipe, args


def build_args_for_config(config: Dict, enable_rewriter: bool) -> Args:
    ckpt_root = DEFAULT_CHECKPOINT_ROOT
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(
            f"找不到权重目录 '{ckpt_root}'，请设置 INFINITYSTAR_CKPT_DIR 指向已下载的模型。"
        )

    model_dir = os.path.join(ckpt_root, config["weights_subdir"])
    vae_path = os.path.join(ckpt_root, "infinitystar_videovae.pth")
    text_encoder_dir = os.path.join(ckpt_root, "text_encoder", "flan-t5-xl-official")
    for path in (model_dir, vae_path, text_encoder_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"缺少必要的权重文件 '{path}'，请确认 InfinityStar 权重已就绪。"
            )

    if enable_rewriter and not os.environ.get("OPEN_API_KEY"):
        raise RuntimeError("已勾选提示词重写，但未设置 OPEN_API_KEY。")

    args = Args()
    args.pn = config["pn"]
    args.fps = 16
    args.video_frames = config["video_frames"]
    args.model_path = model_dir
    args.checkpoint_type = 'torch_shard'
    args.vae_path = vae_path
    args.text_encoder_ckpt = text_encoder_dir
    args.model_type = 'infinity_qwen8b'
    args.text_channels = 2048
    args.dynamic_scale_schedule = 'infinity_elegant_clip20frames_v2'
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = config["image_scale_repetition"]
    args.video_scale_repetition = config["video_scale_repetition"]
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = config["detail_scale_min_tokens"]
    args.semantic_scales = config["semantic_scales"]
    args.max_repeat_times = 10000
    args.enable_rewriter = int(enable_rewriter)

    for attr, value in config.get("arg_overrides", {}).items():
        setattr(args, attr, value)
    return args


def build_interactive_args(enable_rewriter: bool) -> argparse.Namespace:
    ckpt_root = DEFAULT_CHECKPOINT_ROOT
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(
            f"找不到权重目录 '{ckpt_root}'，请设置 INFINITYSTAR_CKPT_DIR 指向已下载的模型。"
        )

    model_dir = os.path.join(ckpt_root, INTERACTIVE_MODEL_SUBDIR)
    vae_path = os.path.join(ckpt_root, "infinitystar_videovae.pth")
    text_encoder_dir = os.path.join(ckpt_root, "text_encoder", "flan-t5-xl-official")
    for path in (model_dir, vae_path, text_encoder_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"缺少必要的权重文件 '{path}'，请确认 InfinityStar 权重已就绪。"
            )

    if enable_rewriter and not os.environ.get("OPEN_API_KEY"):
        raise RuntimeError("已勾选提示词重写，但未设置 OPEN_API_KEY。")

    args = argparse.Namespace(
        pn='0.40M',
        fps=16,
        model_type='infinity_qwen8b',
        h_div_w_template=1.0,
        cache_dir='/dev/shm',
        seed=0,
        bf16=0,
        temporal_slice=0,
        enable_model_cache=0,
        scale_embeds_num=128,
        train_h_div_w_list=[0.571, 1.0],
        steps_per_frame=3,
        context_frames=1000,
        image_batch_size=1,
        video_batch_size=1,
        down_size_limit=340,
        casual_multi_scale=0,
        noise_apply_layers=200,
        noise_apply_requant=1,
        noise_apply_strength=[0.0 for _ in range(100)],
        video_caption_type='tarsier2_caption',
        temporal_compress_rate=4,
        cached_video_frames=81,
        learn_residual=0,
        use_diffloss=0,
        diffusion_batch_mul=0,
        video_fps=16,
        power_value=1.0,
        noise_apply_random_one=0,
        inject_sync=0,
        scales_256=11,
        dummy_text_len_in_seq=0,
        scale_max_token_len=-1,
        same_batch_among_ranks=0,
        use_flex_attn=0,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        sampling_per_bits=1,
    )

    args.model_path = model_dir
    args.vae_path = vae_path
    args.text_encoder_ckpt = text_encoder_dir
    args.checkpoint_type = 'torch_shard'
    args.set_motion_score = -1
    args.min_scale_ind = 3
    args.loop_times_per_scale = 1
    args.global_sid_pe = 0
    args.h_div_w = 0.571
    args.input_noise = 1
    args.use_cfg, args.use_apg, args.cfg, args.apg_norm_threshold = 1, 0, 3, 0.05
    args.diffusion_steps = -1
    args.infinity_diffusion_sample_topk = 1
    args.noise_input = 0
    args.reduce_accumulate_error_method = 'bsc'
    args.map_to_wide_weights = 0
    args.min_duration = -1
    args.use_space_time_quant = 0
    args.use_learnable_dim_proj = 0
    args.semantic_scale_dim = 16
    args.detail_scale_dim = 64
    args.use_prompt_engineering = False
    args.context_from_largest_no = 1
    args.max_repeat_times = 1000
    args.text_channels = 2048
    args.dynamic_scale_schedule = 'infinity_star_interact'
    args.mask_type = 'infinity_star_interact'
    args.semantic_scales = 11
    args.detail_scale_min_tokens = 350
    args.video_frames = 161
    args.max_duration = 10
    args.videovae = 10
    args.vae_type = 64
    args.num_lvl = 2
    args.num_of_label_value = args.num_lvl
    args.semantic_num_lvl = args.num_lvl
    args.semantic_scale_dim = 16
    args.detail_num_lvl = args.num_lvl
    args.detail_scale_dim = 64
    args.use_clipwise_caption = 1
    args.vae_detail = 'discrete_flow_vae'
    args.use_feat_proj = 2
    args.use_fsq_cls_head = 0
    args.rope_type = '4d'
    args.noise_apply_strength = 0.0
    args.task_type = 't2v'
    args.inner_scale_boost = 0
    args.append_duration2caption = 1
    args.n_sampes = 1
    args.duration_resolution = 1
    args.frames_inner_clip = 20
    args.image_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]'
    args.video_scale_repetition = args.image_scale_repetition
    args.taui, args.tauv = 0.5, 0.5
    args.tau = [args.taui] * len(json.loads(args.image_scale_repetition)) + [args.tauv] * len(json.loads(args.video_scale_repetition))
    args.context_interval = 2
    args.simple_text_proj = 1
    args.apply_spatial_patchify = 0
    args.use_two_stage_lfq = 1
    args.fsdp_save_flatten_model = 1
    args.two_gpu_infer = False
    args.other_device = 'cuda:1' if args.two_gpu_infer else 'cuda'
    args.enable_rewriter = int(enable_rewriter)

    return args


class InteractivePipe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        self.vae = load_visual_tokenizer(args).float().to('cuda')
        self.infinity = load_transformer(self.vae, args)
        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
        self.dynamic_resolution_h_w, _ = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
        self.h_div_w_template_list = np.array(list(self.dynamic_resolution_h_w.keys()))
        self.vae_stride = 16
        if args.enable_rewriter:
            self.gpt_model, self.system_prompt = infer_720._init_prompt_rewriter()
        else:
            self.gpt_model, self.system_prompt = None, None


class InteractivePipeCache:
    def __init__(self) -> None:
        self._cache: Dict[int, Tuple[InteractivePipe, argparse.Namespace]] = {}

    def get(self, enable_rewriter: bool) -> Tuple[InteractivePipe, argparse.Namespace]:
        key = int(enable_rewriter)
        if key not in self._cache:
            self._cache[key] = self._build_pipe(enable_rewriter)
        return self._cache[key]

    def _build_pipe(self, enable_rewriter: bool) -> Tuple[InteractivePipe, argparse.Namespace]:
        args = build_interactive_args(enable_rewriter)
        torch.cuda.empty_cache()
        pipe = InteractivePipe(args)
        return pipe, args


def _extract_file_path(file_input) -> Optional[str]:
    if file_input is None:
        return None
    if isinstance(file_input, str) and os.path.exists(file_input):
        return file_input
    if isinstance(file_input, dict):
        candidate = file_input.get("name") or file_input.get("path")
        if candidate and os.path.exists(candidate):
            return candidate
    for attr in ("name", "path"):
        candidate = getattr(file_input, attr, None)
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _sync_reference_preview(file_input) -> Optional[str]:
    return _extract_file_path(file_input)


def _split_prompts(prompt_block: Optional[str]) -> List[str]:
    if not prompt_block:
        return []
    return [line.strip() for line in prompt_block.splitlines() if line.strip()]


def _to_uint8_video(array: np.ndarray) -> np.ndarray:
    video = np.asarray(array)
    if video.dtype == np.uint8:
        return video
    video = np.nan_to_num(video)
    v_min, v_max = float(video.min()), float(video.max())
    if -1.1 <= v_min <= 1.1 and -1.1 <= v_max <= 1.1:
        scaled = ((video + 1.0) * 127.5).clip(0, 255)
    else:
        scaled = (video.clip(0, 1) * 255.0)
    return scaled.astype(np.uint8)


def _load_interactive_examples(max_items: int = 4) -> List[List[object]]:
    examples: List[List[object]] = []
    if not os.path.isdir(INTERACTIVE_TOY_DIR):
        return examples
    for story_id in sorted(os.listdir(INTERACTIVE_TOY_DIR)):
        story_dir = os.path.join(INTERACTIVE_TOY_DIR, story_id)
        if not os.path.isdir(story_dir):
            continue
        prompt_path = os.path.join(story_dir, "prompt.txt")
        if not os.path.isfile(prompt_path):
            continue
        candidate_video = os.path.join(story_dir, "0000_refine_720p.mp4")
        if not os.path.isfile(candidate_video):
            mp4s = [f for f in os.listdir(story_dir) if f.lower().endswith(".mp4")]
            if not mp4s:
                continue
            candidate_video = os.path.join(story_dir, sorted(mp4s)[0])
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_lines = [line.strip() for line in f.readlines() if line.strip()]
        if not prompt_lines:
            continue
        prompts = "\n".join(prompt_lines)
        examples.append([candidate_video, prompts, 41, False])
        if len(examples) >= max_items:
            break
    return examples


PROMPT_REWRITE_PREFIX = (
    "Rewrite the following video descriptions, add more details of the subject and the camera movement to enhance the "
    "quality of the video. Do not use the word 'they' to refer to a single person or object. Concatenate all sentences "
    "together, not present them in paragraphs. Please rewrite with concise and clear language: "
)


def _rewrite_prompt(pipe: InteractivePipe, prompt: str) -> str:
    rewriter = getattr(pipe, "gpt_model", None)
    if rewriter is None:
        return prompt
    try:
        return rewriter(prompt=PROMPT_REWRITE_PREFIX + prompt, system_prompt=pipe.system_prompt)
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"[interactive] 提示词重写失败: {exc}")
        return prompt


def _prepare_reference_context(
    pipe: InteractivePipe,
    args: argparse.Namespace,
    reference_path: str,
) -> Dict[str, object]:
    video = EncodedVideoOpencv(reference_path, os.path.basename(reference_path), num_threads=0)
    duration = getattr(video, 'duration', None)
    if duration is None or duration < INTERACTIVE_SEGMENT_SECONDS:
        raise gr.Error(f"参考视频必须不少于 {INTERACTIVE_SEGMENT_SECONDS} 秒。")
    start_time = max(0.0, duration - INTERACTIVE_SEGMENT_SECONDS)
    raw_video, _ = video.get_clip(start_time, duration, INTERACTIVE_REFERENCE_FRAMES)
    if raw_video is None or len(raw_video) == 0:
        raise gr.Error("参考视频解码失败。")

    h, w, _ = raw_video[0].shape
    aspect_ratio = h / w
    template_values = pipe.h_div_w_template_list
    idx = int(np.argmin(np.abs(aspect_ratio - template_values)))
    template_key = float(template_values[idx])
    schedule_idx = (INTERACTIVE_REFERENCE_FRAMES - 1) // 4 + 1
    scale_schedule = pipe.dynamic_resolution_h_w[template_key][args.pn]['pt2scale_schedule'][schedule_idx]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    scales_in_one_clip = args.first_full_spatial_size_scale_index + 1
    cur_scale_schedule = scale_schedule[scales_in_one_clip:]
    context_info = pipe.get_scale_pack_info(cur_scale_schedule, args.first_full_spatial_size_scale_index, args)
    tgt_h = scale_schedule[-1][1] * pipe.vae_stride
    tgt_w = scale_schedule[-1][2] * pipe.vae_stride

    img_T3HW = [transform(Image.fromarray(frame[:, :, ::-1]), tgt_h, tgt_w) for frame in raw_video]
    img_T3HW = torch.stack(img_T3HW, 0)
    img_bcthw = img_T3HW.permute(1, 0, 2, 3).unsqueeze(0).to('cuda')
    former_clip_features, _, _ = pipe.vae.encode_for_raw_features(img_bcthw, scale_schedule=None, slice=True)
    first_frame_features = former_clip_features[:, :, 0:1]
    resized_reference = np.array([cv2.resize(frame, (tgt_w, tgt_h)) for frame in raw_video], dtype=np.uint8)

    return {
        'reference_frames': resized_reference,
        'former_clip_features': former_clip_features,
        'first_frame_features': first_frame_features,
        'scale_schedule': cur_scale_schedule,
        'context_info': context_info,
    }


def generate_interactive_video(reference_video, prompt_block, seed, enable_rewriter):
    reference_path = _extract_file_path(reference_video)
    if not reference_path or not os.path.exists(reference_path):
        raise gr.Error("请上传一个 MP4 参考视频。")

    prompts = _split_prompts(prompt_block)
    if not prompts:
        raise gr.Error("请至少输入一条提示词（每行一条）。")

    truncated = False
    if len(prompts) > MAX_INTERACTIVE_PROMPTS:
        prompts = prompts[:MAX_INTERACTIVE_PROMPTS]
        truncated = True

    pipe, args = INTERACTIVE_PIPE_CACHE.get(enable_rewriter)
    args.seed = int(seed)

    start_time = time.time()
    ref_context = _prepare_reference_context(pipe, args, reference_path)
    final_segments: List[np.ndarray] = [ref_context['reference_frames']]
    former_clip_features = ref_context['former_clip_features']
    first_frame_features = ref_context['first_frame_features']
    scale_schedule = ref_context['scale_schedule']
    context_info = ref_context['context_info']

    generated_segments = 0
    for idx, raw_prompt in enumerate(prompts):
        prompt_text = _rewrite_prompt(pipe, raw_prompt)
        if args.append_duration2caption:
            prompt_text = f"<<<t={INTERACTIVE_SEGMENT_SECONDS}s>>>" + prompt_text

        clip_seed = int(seed) + idx
        video_tensor, former_clip_features = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt_text,
            negative_prompt="",
            g_seed=clip_seed,
            gt_leak=-1,
            gt_ls_Bl=None,
            cfg_list=args.cfg,
            tau_list=args.tau,
            scale_schedule=scale_schedule,
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=False,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
            mode='second_v_clip',
            former_clip_features=former_clip_features,
            first_frame_features=first_frame_features,
        )
        video_np = _to_uint8_video(video_tensor.cpu().numpy())
        final_segments.append(video_np)
        generated_segments += 1

    if len(final_segments) == 1:
        final_video = final_segments[0]
    else:
        final_video = np.concatenate(final_segments, axis=0)

    with tempfile.TemporaryDirectory(prefix="infstar_interactive_") as tmp_dir:
        intermediate_path = os.path.join(tmp_dir, "interactive_demo.mp4")
        SAVE_VIDEO(final_video, fps=args.fps, save_filepath=intermediate_path)
        fd, final_path = tempfile.mkstemp(prefix="infstar_interactive_", suffix=".mp4")
        os.close(fd)
        shutil.copyfile(intermediate_path, final_path)

    elapsed = time.time() - start_time
    total_duration = INTERACTIVE_SEGMENT_SECONDS * (1 + generated_segments)
    status_bits = [
        f"互动 480p 生成完成，用时 {elapsed:.1f} 秒",
        f"种子={seed}",
        f"续写段数={generated_segments}",
        f"总时长≈{total_duration} 秒",
    ]
    if truncated:
        status_bits.append(f"仅处理前 {MAX_INTERACTIVE_PROMPTS} 条提示")
    status = " | ".join(status_bits)
    return final_path, status

def _maybe_save_image(tmp_dir: str, image) -> Optional[str]:
    if image is None:
        return None
    image = image.convert("RGB")
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="infstar_ref_", suffix=".png", dir=tmp_dir)
    os.close(tmp_fd)
    image.save(tmp_path)
    return tmp_path


def generate_video(
    config_name: str,
    prompt: str,
    seed: int,
    image,
    enable_rewriter: bool,
    duration: Optional[int] = None,
    video_path: Optional[str] = None,
):
    prompt = (prompt or '').strip()
    if not prompt:
        raise gr.Error("提示词不能为空。")

    config = PIPE_CONFIGS[config_name]

    with tempfile.TemporaryDirectory(prefix="infstar_session_") as tmp_dir:
        ref_path = _maybe_save_image(tmp_dir, image)
        pipe, args = PIPE_CACHE.get(config_name, enable_rewriter)
        data = {
            'seed': int(seed),
            'prompt': prompt,
        }
        if ref_path is not None:
            data['image_path'] = ref_path

        if config_name == "480p":
            chosen_duration = int(duration or config["default_duration"])
            if video_path and chosen_duration != max(config["duration_options"]):
                raise gr.Error("若要使用视频续写，请选择 10 秒时长选项。")
            data['duration'] = chosen_duration
            if video_path:
                data['video_path'] = video_path
        result = config['perform_inference'](pipe, data, args)
        intermediate_path = os.path.join(tmp_dir, f'{config_name.lower()}_demo.mp4')
        SAVE_VIDEO(result['output'], fps=args.fps, save_filepath=intermediate_path)
        fd, final_path = tempfile.mkstemp(prefix=f"infstar_{config_name.lower()}_", suffix=".mp4")
        os.close(fd)
        shutil.copyfile(intermediate_path, final_path)

    status = f"{config_name} 生成完成，用时 {result['elapsed_time']:.1f} 秒（种子={seed}）。"
    return final_path, status


def generate_720p(prompt, seed, image, enable_rewriter):
    return generate_video("720p", prompt, seed, image, enable_rewriter)


def generate_480p(prompt, seed, image, enable_rewriter, duration, video_path):
    return generate_video("480p", prompt, seed, image, enable_rewriter, duration=duration, video_path=video_path)


PIPE_CACHE = PipeCache()
INTERACTIVE_PIPE_CACHE = InteractivePipeCache()
INTERACTIVE_EXAMPLES = _load_interactive_examples()


def build_tab_720p():
    config = PIPE_CONFIGS["720p"]
    with gr.Tab(config["tab_label"]):
        gr.Markdown(f"### InfinityStar 8B {config['tab_label']}\n{config['description']}")
        prompt = gr.Textbox(label="提示词", lines=4, placeholder="描述想要生成的画面，例如：傍晚的海边特写...")
        seed = gr.Slider(label="随机种子", minimum=0, maximum=99999, step=1, value=41)
        enable_rewriter = gr.Checkbox(label="启用提示词重写（需设置 OPEN_API_KEY）", value=False)
        image = gr.Image(label="参考图像（可选）", type="pil")
        generate_btn = gr.Button("生成 720p", variant="primary")
        video_output = gr.Video(label="生成视频", autoplay=True)
        status_box = gr.Textbox(label="状态", interactive=False)
        gr.Examples(
            examples=config["examples"],
            inputs=[prompt, seed, image, enable_rewriter],
            label="示例提示词",
        )
        generate_btn.click(
            fn=generate_720p,
            inputs=[prompt, seed, image, enable_rewriter],
            outputs=[video_output, status_box],
        )


def build_tab_480p():
    config = PIPE_CONFIGS["480p"]
    with gr.Tab(config["tab_label"]):
        gr.Markdown(
            "### InfinityStar 8B 480p\n"
            + config["description"]
            + "\n- 可选时长：5 秒（文本/图像生成）或 10 秒（开启视频续写需选此项）。"
        )
        prompt = gr.Textbox(label="提示词", lines=4, placeholder="描述想要生成的画面...")
        seed = gr.Slider(label="随机种子", minimum=0, maximum=99999, step=1, value=41)
        enable_rewriter = gr.Checkbox(label="启用提示词重写（需设置 OPEN_API_KEY）", value=False)
        duration = gr.Radio(
            label="持续时间（秒）",
            choices=config["duration_options"],
            value=config["default_duration"],
            interactive=True,
        )
        image = gr.Image(label="参考图像（可选）", type="pil")
        video_path = gr.File(
            label="续写参考视频（可选，需选择 10 秒）",
            file_types=["video"],
            file_count="single",
        )
        generate_btn = gr.Button("生成 480p", variant="primary")
        video_output = gr.Video(label="生成视频", autoplay=True)
        status_box = gr.Textbox(label="状态", interactive=False)
        gr.Examples(
            examples=config["examples"],
            inputs=[prompt, seed, image, enable_rewriter, duration, video_path],
            label="示例提示词",
        )
        generate_btn.click(
            fn=generate_480p,
            inputs=[prompt, seed, image, enable_rewriter, duration, video_path],
            outputs=[video_output, status_box],
        )


def build_tab_interactive():
    sample_reference = os.path.join(
        "data",
        "interactive_toy_videos",
        "002a061bdbc110ca8fb48e7e0a663c94",
        "0000_refine_720p.mp4",
    )
    default_prompt = (
        "The astronaut approaches a mysterious door\n"
        "The door opens to reveal a lush alien garden"
    )
    example_rows = INTERACTIVE_EXAMPLES or [[sample_reference, default_prompt, 41, False]]
    with gr.Tab("视频续写"):
        gr.Markdown(
            "### InfinityStar 8B 视频续写\n"
            "上传一段 5 秒参考视频并提供多行提示词，模型会按顺序将每条提示扩展约 5 秒剧情，单次最多处理 5 条。"
        )
        reference_video = gr.File(
            label="参考视频（≥5 秒，MP4）",
            file_types=["video"],
            file_count="single",
        )
        reference_preview = gr.Video(label="参考预览", autoplay=False)
        prompts = gr.Textbox(
            label="续写提示词（每行一条）",
            lines=6,
            placeholder="描述接下来发生的事情，例如：他端起咖啡杯 -> 他翻开笔记本",
        )
        seed = gr.Slider(label="随机种子", minimum=0, maximum=99999, step=1, value=41)
        enable_rewriter = gr.Checkbox(label="启用提示词重写（需设置 OPEN_API_KEY）", value=False)
        generate_btn = gr.Button("生成视频续写", variant="primary")
        video_output = gr.Video(label="生成视频", autoplay=True)
        status_box = gr.Textbox(label="状态", interactive=False)
        gr.Examples(
            examples=example_rows,
            inputs=[reference_video, prompts, seed, enable_rewriter],
            label="示例来自 data/interactive_toy_videos",
        )
        generate_btn.click(
            fn=generate_interactive_video,
            inputs=[reference_video, prompts, seed, enable_rewriter],
            outputs=[video_output, status_box],
        )
        reference_video.change(
            fn=_sync_reference_preview,
            inputs=reference_video,
            outputs=reference_preview,
        )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="InfinityStar 演示") as demo:
        gr.Markdown(
            "## InfinityStar 体验面板\n"
            "选择不同标签即可运行官方 720p、480p 以及互动 480p 推理流程，可按需上传参考素材，"
            "若已配置 OPEN_API_KEY 还可启用提示词重写。"
        )
        with gr.Tabs():
            build_tab_720p()
            build_tab_480p()
            build_tab_interactive()
    return demo


def main():
    demo = build_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )


if __name__ == "__main__":
    main()
