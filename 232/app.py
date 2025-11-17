import argparse
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request

import gradio as gr
import soundfile as sf
import torch
from PIL import Image as PIL_Image, ImageDraw

from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

DEFAULT_CHECKPOINT = "checkpoints/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_CAPTIONER_CHECKPOINT = "checkpoints/Qwen3-Omni-30B-A3B-Captioner"
DEFAULT_THINKING_CHECKPOINT = "checkpoints/Qwen3-Omni-30B-A3B-Thinking"
DEFAULT_PROMPT = "Give the detailed description of the audio."
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 20
DEFAULT_MAX_NEW_TOKENS = 1024
VOICE_CHOICES = ["Chelsie", "Ethan", "Aiden"]


@dataclass(frozen=True)
class SampleAsset:
    label: str
    type: str
    url: str
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class TaskConfig:
    slug: str
    title: str
    description: str
    modalities: Tuple[str, ...]
    default_prompt: str = ""
    show_prompt_box: bool = True
    prompt_required: bool = False
    allow_system_prompt: bool = False
    default_system_prompt: str = ""
    checkpoint_key: str = "instruct"
    sample_assets: Tuple[SampleAsset, ...] = ()
    supports_audio_reply: bool = False
    default_want_audio: bool = False
    default_voice: str = "Ethan"
    use_audio_in_video: bool = True
    notes: Optional[str] = None
    generation_defaults: Dict[str, float] = field(default_factory=dict)
    show_grounding_boxes: bool = False


MODEL_CACHE: Dict[Tuple[str, bool, str], Tuple[Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor]] = {}


def audio_sample(label: str, url: str, prompt: Optional[str] = None, system_prompt: Optional[str] = None) -> SampleAsset:
    return SampleAsset(label=label, type="audio", url=url, prompt=prompt, system_prompt=system_prompt)


def image_sample(label: str, url: str, prompt: Optional[str] = None, system_prompt: Optional[str] = None) -> SampleAsset:
    return SampleAsset(label=label, type="image", url=url, prompt=prompt, system_prompt=system_prompt)


def video_sample(label: str, url: str, prompt: Optional[str] = None, system_prompt: Optional[str] = None) -> SampleAsset:
    return SampleAsset(label=label, type="video", url=url, prompt=prompt, system_prompt=system_prompt)


def _list_local_model_paths(base_dir: str = "checkpoints") -> List[str]:
    base = Path(base_dir)
    if not base.exists():
        return []
    candidates: List[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if (child / "config.json").exists():
            candidates.append(str(child))
    return candidates


def _refresh_model_dropdown(current_value: Optional[str] = None):
    choices = _list_local_model_paths()
    value = current_value if current_value else (choices[0] if choices else None)
    if current_value and current_value not in choices:
        value = current_value
    return gr.Dropdown.update(choices=choices, value=value)


def _release_models():
    released = len(MODEL_CACHE)
    if released == 0:
        return released
    for model, _ in list(MODEL_CACHE.values()):
        try:
            model.cpu()
        except Exception:
            pass
        del model
    MODEL_CACHE.clear()
    torch.cuda.empty_cache()
    return released


def _switch_model(path: str, args):
    target = (path or "").strip()
    if not target:
        raise gr.Error("请选择或输入模型路径。")
    if target == args.checkpoint_path and MODEL_CACHE:
        return f"模型已加载：{target}"
    _release_models()
    args.checkpoint_path = target
    try:
        _get_model_and_processor(target, args)
    except Exception as exc:
        _release_models()
        raise gr.Error(f"加载模型失败：{exc}")
    return f"已成功加载模型：{target}"


SYSTEM_TRANSLATOR = (
    "You are a virtual voice assistant with no gender or age.\n"
    "Interact naturally, keep replies under 50 words, and only output the spoken response.\n"
    "Mirror the user's language unless they request otherwise."
)
SYSTEM_DRAW = (
    "You are a voice assistant. Answer user questions without directly narrating the video."
)
SYSTEM_ROMANTIC = (
    "You are a romantic and artistic AI. Use metaphors and poetry when speaking, respond concisely."
)
SYSTEM_BEIJING = "你是一个北京大爷，说话幽默，使用地道北京话。"
SYSTEM_FUNCTION_CALL = (
    "You may call one or more functions to assist with the user query.\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
    "{'type': 'function', 'function': {'name': 'web_search', 'description': 'Utilize the web search engine to retrieve relevant information based on multiple queries.', 'parameters': {'type': 'object', 'properties': {'queries': {'type': 'array', 'items': {'type': 'string', 'description': 'The search query.'}, 'description': 'The list of search queries.'}}, 'required': ['queries']}}}"
    "{'type': 'function', 'function': {'name': 'car_ac_control', 'description': \"Control the vehicle's air conditioning system to turn it on/off and set the target temperature\", 'parameters': {'type': 'object', 'properties': {'temperature': {'type': 'number', 'description': 'Target set temperature in Celsius degrees'}, 'ac_on': {'type': 'boolean', 'description': 'Air conditioning status (true=on, false=off)'}}, 'required': ['temperature', 'ac_on']}}}"
    "</tools>\n"
    "For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> tags."
)


TASK_CONFIGS: List[TaskConfig] = [
    TaskConfig(
        slug="audio_caption",
        title="Audio Caption",
        description="Upload or record audio and let Qwen3-Omni describe it in detail.",
        modalities=("audio",),
        default_prompt=DEFAULT_PROMPT,
        supports_audio_reply=True,
        default_want_audio=False,
        sample_assets=(
            audio_sample("Caption 1", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/caption1.mp3"),
            audio_sample("Caption 2", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/caption2.mp3"),
            audio_sample("Caption 3", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/caption3.mp3"),
        ),
    ),
    TaskConfig(
        slug="audio_visual_dialogue",
        title="Audio-Visual Dialogue",
        description="Chat with the assistant using audio or video inputs and optionally receive spoken replies.",
        modalities=("audio", "video"),
        allow_system_prompt=True,
        default_system_prompt=SYSTEM_TRANSLATOR,
        supports_audio_reply=True,
        default_want_audio=True,
        sample_assets=(
            audio_sample("Audio translation", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/translate_to_chinese.wav", system_prompt=SYSTEM_TRANSLATOR),
            video_sample(
                "Sketch board Q&A",
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/draw.mp4",
                system_prompt=SYSTEM_DRAW,
            ),
        ),
    ),
    TaskConfig(
        slug="audio_visual_interaction",
        title="Audio-Visual Interaction",
        description="Drive free-form conversations grounded in uploaded audio or video clips.",
        modalities=("audio", "video"),
        allow_system_prompt=True,
        default_system_prompt="",
        supports_audio_reply=True,
        default_want_audio=True,
        sample_assets=(
            audio_sample("Chat - EN", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction1.mp3"),
            video_sample("Drive-thru", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction2.mp4"),
            audio_sample("Romantic POI", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction3.mp3", system_prompt=SYSTEM_ROMANTIC),
            video_sample("Beijing Humor", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction4.mp4", system_prompt=SYSTEM_BEIJING),
        ),
    ),
    TaskConfig(
        slug="audio_visual_question",
        title="Audio-Visual Question",
        description="Ask detailed questions about multimodal videos and get grounded answers.",
        modalities=("video",),
        default_prompt="What was the first sentence the boy said when he met the girl?",
        sample_assets=(
            video_sample("Story QA", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual.mp4", prompt="What was the first sentence the boy said when he met the girl?"),
            video_sample(
                "Finance quiz",
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual2.mp4",
                prompt=(
                    "Question: What narrative purpose do the question marks above Will's head serve when they first appear?\n"
                    "Choices: ['A. To show Will\'s confusion about SEC regulations', 'B. To visualize Will analyzing market trends',\n"
                    "'C. To demonstrate investor demand for his expertise', 'D. To indicate SEC scrutiny of his fund proposal']\n"
                    "Please give your answer."
                ),
            ),
        ),
    ),
    TaskConfig(
        slug="audio_function_call",
        title="Audio Function Call",
        description="Test tool-calling instructions using spoken commands.",
        modalities=("audio",),
        show_prompt_box=False,
        allow_system_prompt=True,
        default_system_prompt=SYSTEM_FUNCTION_CALL,
        sample_assets=(
            audio_sample("Smart cabin", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/functioncall_case.wav", system_prompt=SYSTEM_FUNCTION_CALL),
        ),
        generation_defaults={"temperature": 0.2, "top_p": 0.3, "top_k": 10},
    ),
    TaskConfig(
        slug="image_math",
        title="Image Math",
        description="Reason about diagrams or charts that include math or physics prompts.",
        modalities=("image",),
        checkpoint_key="thinking",
        default_prompt="Describe what the chart represents and answer the question.",
        sample_assets=(
            image_sample(
                "Sprinkler problem",
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/5195.jpg",
                prompt=("The 3-arm lawn sprinkler receives 20°C water at 2.7 m^3/hr. If collar friction is neglected, what is the steady rotation rate?"
                "\nOptions: A.317 rev/min B.414 rev/min C.400 rev/min D.NaN"),
            ),
            image_sample(
                "J(theta) puzzle",
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/4181.jpg",
                prompt=(
                    "Suppose we have m=3 training samples (plot shown). Hypothesis h_θ(x)=θ₁x. What is J(0)?"
                    "\nOptions: A.0 B.1/6 C.1 D.14/6"
                ),
            ),
        ),
    ),
    TaskConfig(
        slug="image_question",
        title="Image Question",
        description="General visual question answering for uploaded images.",
        modalities=("image",),
        default_prompt="What style does this image depict?",
        sample_assets=(
            image_sample("Interior style", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/2621.jpg", prompt="What style does this image depict?"),
            image_sample("Next event", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/2233.jpg", prompt="Based on this image, what do you think will happen next?"),
            image_sample("IQ pattern", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/val_IQ_Test_113.jpg", prompt="Identify the picture that follows the pattern and answer directly."),
        ),
    ),
    TaskConfig(
        slug="object_grounding",
        title="Object Grounding",
        description="Locate objects in images and visualize bounding boxes from model output.",
        modalities=("image",),
        default_prompt="Locate the object: bird.",
        prompt_required=True,
        sample_assets=(
            image_sample("Bird", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/grounding1.jpeg", prompt="Locate the object: bird."),
            image_sample(
                "Motorcyclist",
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/grounding2.jpg",
                prompt="Locate the object: A person riding a motorcycle while wearing a helmet.",
            ),
        ),
        show_grounding_boxes=True,
    ),
    TaskConfig(
        slug="ocr",
        title="OCR",
        description="Extract multilingual text directly from images.",
        modalities=("image",),
        default_prompt="请提取图片中的文字。",
        sample_assets=(
            image_sample("Menu", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/ocr2.jpeg", prompt="请提取图片中的文字。"),
            image_sample("Sign", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/ocr1.jpeg", prompt="Extract the text from the image."),
        ),
    ),
    TaskConfig(
        slug="mixed_audio_analysis",
        title="Mixed Audio Analysis",
        description="Classify speakers, sound effects, and instruments inside short clips.",
        modalities=("audio",),
        default_prompt="Determine all sound events in the clip.",
        sample_assets=(
            audio_sample("CN speaker", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/mixed_audio1.mp3", prompt="判断说话人的国籍和性别，并告诉我音频里出现的音效是什么？"),
            audio_sample("SFX mix", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/mixed_audio2.mp3", prompt="Determine which sound effects and musical instruments are present in the audio."),
        ),
    ),
    TaskConfig(
        slug="music_analysis",
        title="Music Analysis",
        description="Describe genres, instruments, and emotions for pieces of music.",
        modalities=("audio",),
        default_prompt="请分析这是什么风格的音乐？",
        sample_assets=(
            audio_sample("风格判断", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/音乐风格-调性.mp3", prompt="请分析这是什么风格的音乐？"),
            audio_sample("37573", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/37573.mp3", prompt="Describe the style, rhythm, dynamics, emotions, instruments, and possible scenarios."),
            audio_sample("353349", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/353349.mp3", prompt="Write an appreciative description, identify genre and analyze instrument collaboration."),
        ),
    ),
    TaskConfig(
        slug="omni_captioner",
        title="Omni Captioner",
        description="Use the Qwen3-Omni-30B-A3B-Captioner checkpoint for autonomous audio captions.",
        modalities=("audio",),
        show_prompt_box=False,
        checkpoint_key="captioner",
        notes="该模型仅接收单段音频输入并输出文本描述。",
        sample_assets=(
            audio_sample("Captioner Case 1", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/captioner-case1.wav"),
            audio_sample("Captioner Case 2", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/captioner-case2.wav"),
            audio_sample("Captioner Case 3", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/captioner-case3.wav"),
        ),
    ),
    TaskConfig(
        slug="sound_analysis",
        title="Sound Analysis",
        description="Classify environment sounds and predict likely scenarios.",
        modalities=("audio",),
        default_prompt="What happened in the audio?",
        sample_assets=(
            audio_sample("Sound 1", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/sound1.wav", prompt="What happened in the audio?"),
            audio_sample("Sound 2", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/sound2.mp3", prompt="What is this sound? In what situation might it occur?"),
            audio_sample("Sound 3", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/sound3.mp3", prompt="Guess where I am?"),
        ),
    ),
    TaskConfig(
        slug="speech_recognition",
        title="Speech Recognition",
        description="Transcribe multilingual speech to plain text.",
        modalities=("audio",),
        default_prompt="Transcribe this speech into text.",
        sample_assets=(
            audio_sample("ASR ZH", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav", prompt="请将这段中文语音转换为纯文本。"),
            audio_sample("ASR EN", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_en.wav", prompt="Transcribe the English audio into text."),
            audio_sample("ASR FR", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_fr.wav", prompt="Transcribe the French audio into text."),
        ),
        generation_defaults={"temperature": 0.2, "top_p": 0.5, "top_k": 5},
    ),
    TaskConfig(
        slug="speech_translation",
        title="Speech Translation",
        description="Translate spoken language into another language in one step.",
        modalities=("audio",),
        default_prompt="Listen to the speech and translate it.",
        sample_assets=(
            audio_sample("ZH→EN", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav", prompt="Listen to the provided Chinese speech and produce a translation in English text."),
            audio_sample("EN→ZH", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_en.wav", prompt="Listen to the provided English speech and produce a translation in Chinese text."),
            audio_sample("FR→EN", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_fr.wav", prompt="Listen to the provided French speech and produce a translation in English text."),
        ),
        generation_defaults={"temperature": 0.2, "top_p": 0.5, "top_k": 5},
    ),
    TaskConfig(
        slug="video_description",
        title="Video Description",
        description="Describe the main content of short videos (visual focus).",
        modalities=("video",),
        default_prompt="Describe the video.",
        use_audio_in_video=False,
        sample_assets=(
            video_sample("Video 1", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/video1.mp4", prompt="Describe the video."),
        ),
    ),
    TaskConfig(
        slug="video_navigation",
        title="Video Navigation",
        description="Ask spatial/navigation questions about a given video.",
        modalities=("video",),
        default_prompt="If I want to stop at the window, which direction should I take?",
        use_audio_in_video=False,
        sample_assets=(
            video_sample(
                "Hallway route",
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/video2.mp4",
                prompt="If I want to stop at the window, which direction should I take?",
            ),
        ),
    ),
    TaskConfig(
        slug="video_scene_transition",
        title="Video Scene Transition",
        description="Summarize how scenes change throughout a clip.",
        modalities=("video",),
        default_prompt="How do the scenes in the video change?",
        use_audio_in_video=False,
        sample_assets=(
            video_sample("Scene cuts", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/video4.mp4", prompt="How the scenes in the video change?"),
        ),
    ),
]

TASK_LOOKUP = {cfg.slug: cfg for cfg in TASK_CONFIGS}
SAMPLE_LIBRARY: Dict[str, Dict[str, SampleAsset]] = {
    cfg.slug: {asset.label: asset for asset in cfg.sample_assets} for cfg in TASK_CONFIGS
}


def _ensure_initializer_range(target_cfg, *fallback_cfgs):
    if target_cfg is None:
        return
    if getattr(target_cfg, "initializer_range", None) is not None:
        return
    for cfg in fallback_cfgs:
        if cfg is None:
            continue
        value = getattr(cfg, "initializer_range", None)
        if value is not None:
            setattr(target_cfg, "initializer_range", value)
            return
    setattr(target_cfg, "initializer_range", 0.02)


def _prepare_config(checkpoint_path: str) -> Qwen3OmniMoeConfig:
    config = Qwen3OmniMoeConfig.from_pretrained(checkpoint_path)
    if getattr(config, "initializer_range", None) is None:
        setattr(config, "initializer_range", 0.02)
    talker_cfg = getattr(config, "talker_config", None)
    thinker_cfg = getattr(config, "thinker_config", None)
    talker_text_cfg = getattr(talker_cfg, "text_config", None) if talker_cfg else None
    code2wav_cfg = getattr(config, "code2wav_config", None)

    _ensure_initializer_range(talker_text_cfg, thinker_cfg, config)
    _ensure_initializer_range(talker_cfg, talker_text_cfg, thinker_cfg, config)
    _ensure_initializer_range(code2wav_cfg, talker_cfg, thinker_cfg, config)

    if getattr(config, "tie_weights_keys", None):
        print("[app] Info: disabling tie_weights_keys for local checkpoint compatibility.")
        config.tie_weights_keys = []
    return config


def _dtype_from_flag(flag: str):
    if flag == "auto":
        return "auto"
    if flag == "bfloat16":
        return torch.bfloat16
    if flag == "float16":
        return torch.float16
    if flag == "float32":
        return torch.float32
    return None


def _resolve_checkpoint_path(key: str, args) -> str:
    if key == "captioner":
        return args.captioner_checkpoint_path or args.checkpoint_path
    if key == "thinking":
        return args.thinking_checkpoint_path or args.checkpoint_path
    return args.checkpoint_path


def _get_model_and_processor(checkpoint_path: str, args):
    dtype_key = args.dtype
    cache_key = (checkpoint_path, args.flash_attn2, dtype_key)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    config = _prepare_config(checkpoint_path)
    torch_dtype = _dtype_from_flag(args.dtype)
    dtype_kwargs = {}
    if torch_dtype is not None:
        dtype_kwargs["torch_dtype"] = torch_dtype

    attn_impl = "flash_attention_2" if args.flash_attn2 else "eager"

    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            checkpoint_path,
            config=config,
            device_map="auto",
            attn_implementation=attn_impl,
            **dtype_kwargs,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        if args.flash_attn2:
            print(f"[app] Warning: FlashAttention 2 unavailable ({exc}). Falling back to eager.")
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                checkpoint_path,
                config=config,
                device_map="auto",
                attn_implementation="eager",
                **dtype_kwargs,
            )
        else:
            raise
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(checkpoint_path)
    MODEL_CACHE[cache_key] = (model, processor)
    return model, processor


def _download_asset(url: str) -> str:
    suffix = Path(url).suffix or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin")
    with request.urlopen(url) as resp, open(tmp.name, "wb") as fout:
        fout.write(resp.read())
    return tmp.name


def _save_audio_file(audio_tensor: Optional[torch.Tensor], sample_rate: int = 24000) -> Optional[str]:
    if audio_tensor is None:
        return None
    array = audio_tensor.reshape(-1).detach().cpu().numpy()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, array, samplerate=sample_rate)
    return tmp.name


def _extract_json_from_string(text: str) -> str:
    start_indices = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    end_indices = [idx for idx in (text.rfind("}"), text.rfind("]")) if idx != -1]
    if not start_indices or not end_indices:
        return text
    start = min(start_indices)
    end = max(end_indices)
    return text[start : end + 1]


def _draw_bounding_boxes(image_path: str, response_text: str) -> Optional[str]:
    if not image_path:
        return None
    try:
        image = PIL_Image.open(image_path).convert("RGB")
    except Exception:
        return None
    clean_json = _extract_json_from_string(response_text)
    try:
        locations = json.loads(clean_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(locations, list):
        return None
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for loc in locations:
        bbox = loc.get("bbox_2d") if isinstance(loc, dict) else None
        if not bbox or len(bbox) != 4:
            continue
        x1 = (bbox[0] / 1000) * width
        y1 = (bbox[1] / 1000) * height
        x2 = (bbox[2] / 1000) * width
        y2 = (bbox[3] / 1000) * height
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(tmp.name)
    return tmp.name


def _compose_messages(
    audio_path: Optional[str],
    image_path: Optional[str],
    video_path: Optional[str],
    prompt_text: str,
    system_prompt: str,
    cfg: TaskConfig,
) -> List[Dict]:
    prompt_text = (prompt_text or "").strip()
    system_prompt = (system_prompt or "").strip() or cfg.default_system_prompt
    if cfg.show_prompt_box:
        if not prompt_text and cfg.default_prompt:
            prompt_text = cfg.default_prompt
    else:
        prompt_text = prompt_text or cfg.default_prompt
    if cfg.prompt_required and not prompt_text:
        raise gr.Error("该任务需要填写提示词。")

    contents: List[Dict[str, str]] = []
    if "image" in cfg.modalities and image_path:
        contents.append({"type": "image", "image": image_path})
    if "video" in cfg.modalities and video_path:
        contents.append({"type": "video", "video": video_path})
    if "audio" in cfg.modalities and audio_path:
        contents.append({"type": "audio", "audio": audio_path})
    if prompt_text:
        contents.append({"type": "text", "text": prompt_text})
    if not contents:
        raise gr.Error("请至少提供一个输入（音频/视频/图片或文本）。")

    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": contents})
    return messages


def _align_input_dtypes(model, inputs):
    model_dtype = None
    for param in model.parameters():
        model_dtype = param.dtype
        break
    if model_dtype is None:
        return inputs
    for key, value in inputs.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            inputs[key] = value.to(model_dtype)
    return inputs


def _generate_response(
    cfg: TaskConfig,
    args,
    audio_path: Optional[str],
    image_path: Optional[str],
    video_path: Optional[str],
    prompt_text: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    top_k: float,
    max_new_tokens: float,
    want_audio: bool,
    voice_choice: str,
):
    messages = _compose_messages(audio_path, image_path, video_path, prompt_text, system_prompt, cfg)
    checkpoint_path = _resolve_checkpoint_path(cfg.checkpoint_key, args)
    model, processor = _get_model_and_processor(checkpoint_path, args)

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=cfg.use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        use_audio_in_video=cfg.use_audio_in_video,
    )
    inputs = inputs.to(model.device)
    inputs = _align_input_dtypes(model, inputs)

    gen_kwargs = dict(
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=int(max_new_tokens),
        thinker_do_sample=True,
        thinker_temperature=float(temperature),
        thinker_top_p=float(top_p),
        thinker_top_k=int(top_k),
        speaker=voice_choice or cfg.default_voice,
        use_audio_in_video=cfg.use_audio_in_video,
        return_audio=bool(want_audio and cfg.supports_audio_reply),
    )

    with torch.inference_mode():
        text_ids, audio = model.generate(**inputs, **gen_kwargs)

    response_text = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    audio_file = _save_audio_file(audio) if audio is not None else None
    grounding_image = _draw_bounding_boxes(image_path, response_text) if cfg.show_grounding_boxes else None
    outputs = [response_text]
    if cfg.supports_audio_reply:
        outputs.append(audio_file)
    if cfg.show_grounding_boxes:
        outputs.append(grounding_image)
    return tuple(outputs)


def _load_sample(task_slug: str, label: str):
    library = SAMPLE_LIBRARY.get(task_slug)
    if not library:
        raise gr.Error("该任务没有预设示例。")
    if not label:
        raise gr.Error("请选择一个示例。")
    asset = library.get(label)
    if asset is None:
        raise gr.Error("未找到该示例。")
    file_path = _download_asset(asset.url)
    audio_val = gr.update()
    image_val = gr.update()
    video_val = gr.update()
    if asset.type == "audio":
        audio_val = file_path
    elif asset.type == "image":
        image_val = file_path
    elif asset.type == "video":
        video_val = file_path
    prompt_val = asset.prompt if asset.prompt is not None else gr.update()
    system_val = asset.system_prompt if asset.system_prompt is not None else gr.update()
    return audio_val, image_val, video_val, prompt_val, system_val


def _build_tab(cfg: TaskConfig, args):
    with gr.Tab(cfg.title):
        gr.Markdown(f"### {cfg.title}\n{cfg.description}")
        if cfg.notes:
            gr.Markdown(f"> {cfg.notes}")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="音频输入", sources=["upload", "microphone"], type="filepath") if "audio" in cfg.modalities else None
                image_input = gr.Image(label="图片输入", sources=["upload"], type="filepath") if "image" in cfg.modalities else None
                video_input = gr.Video(label="视频输入") if "video" in cfg.modalities else None

                prompt_box = (
                    gr.Textbox(label="提示词", value=cfg.default_prompt, lines=3)
                    if cfg.show_prompt_box
                    else None
                )
                system_box = (
                    gr.Textbox(label="系统提示 (可选)", value=cfg.default_system_prompt, lines=3)
                    if cfg.allow_system_prompt
                    else None
                )

                with gr.Accordion("生成参数", open=False):
                    temp_slider = gr.Slider(0.1, 2.0, step=0.05, label="Temperature", value=cfg.generation_defaults.get("temperature", DEFAULT_TEMPERATURE))
                    top_p_slider = gr.Slider(0.1, 1.0, step=0.05, label="Top P", value=cfg.generation_defaults.get("top_p", DEFAULT_TOP_P))
                    top_k_slider = gr.Slider(1, 128, step=1, label="Top K", value=int(cfg.generation_defaults.get("top_k", DEFAULT_TOP_K)))
                    max_tokens_slider = gr.Slider(64, 8192, step=64, label="Max new tokens", value=int(cfg.generation_defaults.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)))
                    if cfg.supports_audio_reply:
                        want_audio = gr.Checkbox(label="返回语音回复", value=cfg.default_want_audio)
                        voice_choice = gr.Dropdown(VOICE_CHOICES, value=cfg.default_voice, label="语音风格")
                    else:
                        want_audio = gr.State(False)
                        voice_choice = gr.State(cfg.default_voice)

                sample_dropdown = None
                sample_button = None
                if SAMPLE_LIBRARY.get(cfg.slug):
                    with gr.Accordion("示例素材", open=False):
                        sample_dropdown = gr.Dropdown(
                            choices=list(SAMPLE_LIBRARY[cfg.slug].keys()),
                            label="选择示例",
                        )
                        sample_button = gr.Button("加载示例")

                run_button = gr.Button("开始生成", variant="primary")
                clear_button = gr.Button("清空")

            with gr.Column(scale=1):
                text_output = gr.Textbox(label="模型回答", lines=12)
                audio_output = gr.Audio(label="语音回答", type="filepath") if cfg.supports_audio_reply else None
                grounding_output = gr.Image(label="标注结果") if cfg.show_grounding_boxes else None

        def _inference(audio, image, video, prompt, system, temp, top_p, top_k, max_tokens, wa, vc):
            return _generate_response(
                cfg,
                args,
                audio,
                image,
                video,
                prompt,
                system,
                temp,
                top_p,
                top_k,
                max_tokens,
                wa,
                vc,
            )

        run_inputs = [
            audio_input or gr.State(None),
            image_input or gr.State(None),
            video_input or gr.State(None),
            prompt_box or gr.State(""),
            system_box or gr.State(""),
            temp_slider,
            top_p_slider,
            top_k_slider,
            max_tokens_slider,
            want_audio,
            voice_choice,
        ]
        run_outputs = [text_output]
        if cfg.supports_audio_reply:
            run_outputs.append(audio_output)
        if cfg.show_grounding_boxes:
            run_outputs.append(grounding_output)

        run_button.click(fn=_inference, inputs=run_inputs, outputs=run_outputs)

        clear_outputs: List = []
        clear_values: List = []
        if audio_input:
            clear_outputs.append(audio_input)
            clear_values.append(None)
        if image_input:
            clear_outputs.append(image_input)
            clear_values.append(None)
        if video_input:
            clear_outputs.append(video_input)
            clear_values.append(None)
        if prompt_box:
            clear_outputs.append(prompt_box)
            clear_values.append(cfg.default_prompt)
        if system_box:
            clear_outputs.append(system_box)
            clear_values.append(cfg.default_system_prompt)
        clear_outputs.extend([temp_slider, top_p_slider, top_k_slider, max_tokens_slider])
        clear_values.extend([
            cfg.generation_defaults.get("temperature", DEFAULT_TEMPERATURE),
            cfg.generation_defaults.get("top_p", DEFAULT_TOP_P),
            int(cfg.generation_defaults.get("top_k", DEFAULT_TOP_K)),
            int(cfg.generation_defaults.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        ])
        if cfg.supports_audio_reply:
            clear_outputs.extend([audio_output, want_audio, voice_choice])
            clear_values.extend([None, cfg.default_want_audio, cfg.default_voice])
        else:
            if audio_output:
                clear_outputs.append(audio_output)
                clear_values.append(None)
        if cfg.show_grounding_boxes:
            clear_outputs.append(grounding_output)
            clear_values.append(None)

        if clear_outputs:
            clear_button.click(fn=lambda: tuple(clear_values), outputs=clear_outputs)

        if sample_button and sample_dropdown:
            sample_button.click(
                fn=lambda label: _load_sample(cfg.slug, label),
                inputs=sample_dropdown,
                outputs=[
                    audio_input or gr.State(None),
                    image_input or gr.State(None),
                    video_input or gr.State(None),
                    prompt_box or gr.State(""),
                    system_box or gr.State(""),
                ],
            )


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Qwen3-Omni cookbooks")
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT, help="Instruct 模型路径或权重名称")
    parser.add_argument("--captioner-checkpoint-path", type=str, default=DEFAULT_CAPTIONER_CHECKPOINT, help="Captioner 模型路径")
    parser.add_argument("--thinking-checkpoint-path", type=str, default=DEFAULT_THINKING_CHECKPOINT, help="Thinking 模型路径")
    parser.add_argument("--flash-attn2", action="store_true", help="启用 FlashAttention2 (如果可用)")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="模型权重 dtype",
    )
    parser.add_argument("--share", action="store_true", help="创建可分享链接")
    parser.add_argument("--inbrowser", action="store_true", help="自动打开浏览器")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio 端口")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="服务地址")
    return parser


def main():
    args = _build_arg_parser().parse_args()

    initial_choices = _list_local_model_paths()
    initial_value = args.checkpoint_path if args.checkpoint_path in initial_choices else (initial_choices[0] if initial_choices else args.checkpoint_path)

    def _handle_load(selected_path, manual_path):
        chosen = (manual_path or "").strip() or selected_path or args.checkpoint_path
        message = _switch_model(chosen, args)
        return f"当前模型：{args.checkpoint_path}\n\n{message}"

    def _handle_refresh(current_value):
        update = _refresh_model_dropdown(current_value)
        choices = update.get("choices") or []
        value = update.get("value")
        summary = f"已检测到 {len(choices)} 个模型目录。当前选择：{value or '（无）'}"
        return update, summary

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌐 Qwen3-Omni Cookbook WebUI")
        gr.Markdown(
            "选择任意标签页，上传音频/视频/图像，即可复现实验手册中的示例。所有任务共用同一后端模型，支持本地权重。"
        )

        with gr.Accordion("模型管理", open=False):
            model_dropdown = gr.Dropdown(
                label="检测到的模型目录",
                choices=initial_choices,
                value=initial_value,
                allow_custom_value=False,
            )
            manual_input = gr.Textbox(label="或手动输入模型路径", value=args.checkpoint_path)
            with gr.Row():
                load_button = gr.Button("加载/切换模型", variant="primary")
                refresh_button = gr.Button("刷新列表")
            status_markdown = gr.Markdown(value=f"当前模型：{args.checkpoint_path}")

        load_button.click(
            fn=_handle_load,
            inputs=[model_dropdown, manual_input],
            outputs=status_markdown,
        )
        refresh_button.click(
            fn=_handle_refresh,
            inputs=model_dropdown,
            outputs=[model_dropdown, status_markdown],
        )

        for cfg in TASK_CONFIGS:
            _build_tab(cfg, args)
    print(
        f"[app] Launching Gradio server on {args.server_name}:{args.server_port} (share={args.share})..."
    )
    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()