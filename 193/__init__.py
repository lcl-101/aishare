"""Main entry point for the olmoasr package."""

import io
import os
import re
from pathlib import Path
import urllib.request
import urllib.error
from typing import Optional, Union
import torch
from olmoasr import (
    model,
    inf_model,
    preprocess,
    utils,
)

# from whisper import audio, decoding, transcribe
from whisper import audio, decoding
from olmoasr import transcribe
from olmoasr.model import ModelDimensions, OLMoASR
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim

MODEL2LINK = {
    "tiny": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-tiny.en.pt",
    "base": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-base.en.pt",
    "small": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-small.en.pt",
    "medium": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-medium-v2.en.pt",
    "large": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-large.en.pt",
    "large-v2": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-large.en-v2.pt",
}


def _get_cache_dir(download_root: Optional[str] = None) -> Path:
    """Get the cache directory for storing downloaded models."""
    if download_root is not None:
        cache_dir = Path(download_root).expanduser().resolve()
    else:
        cache_dir = Path.home() / ".cache" / "olmoasr"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_model(
    url: str, model_name: str, download_root: Optional[str] = None
) -> str:
    """
    Download a model from a URL and cache it locally.

    Parameters
    ----------
    url : str
        URL to download the model from
    model_name : str
        Name of the model for caching
    download_root : str, optional
        Path to download the model files; by default, it uses "~/.cache/olmoasr"

    Returns
    -------
    str
        Path to the downloaded model file
    """
    cache_dir = _get_cache_dir(download_root)
    filename = f"OLMoASR-{model_name}.pt"
    cache_path = cache_dir / filename

    # Return cached file if it exists
    if cache_path.exists():
        print(f"Using cached model: {cache_path}")
        return str(cache_path)

    print(f"Downloading {model_name} model from {url}...")
    print(f"Saving to: {cache_path}")

    try:
        # Download with progress indication
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rDownloading... {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, cache_path, reporthook=progress_hook)
        print(f"\nModel downloaded successfully: {cache_path}")
        return str(cache_path)

    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download model from {url}: {e}")
    except Exception as e:
        # Clean up partial download
        if cache_path.exists():
            cache_path.unlink()
        raise RuntimeError(f"Error downloading model: {e}")


# should add more features (loading in model checkpoints by identifiers with dictionary of checkpoint paths)
def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: Optional[str] = None,
    inference: bool = False,
    in_memory: bool = False,
) -> OLMoASR:
    """
    Load a OLMoASR model

    Parameters
    ----------
    name : str
        one of the official model names listed in MODEL2LINK, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root : str, optional
        path to download the model files; by default, it uses "~/.cache/olmoasr"
    inference : bool
        whether to load the inference version of the model
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : OLMoASR
        The OLMoASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if name is a model identifier in MODEL2LINK
    if name in MODEL2LINK:
        checkpoint_file = _download_model(MODEL2LINK[name], name, download_root)
    elif os.path.isfile(name):
        checkpoint_file = name
    else:
        raise ValueError(
            f"Model '{name}' not found. Available models: {list(MODEL2LINK.keys())}"
        )

    # Load model weights into memory if requested
    if in_memory:
        with open(checkpoint_file, "rb") as f:
            checkpoint_file = f.read()

    alignment_heads = None

    with (io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")) as fp:
        checkpoint = torch.load(fp, map_location=device, weights_only=False)

    # Clean up if we loaded into memory
    if in_memory:
        del checkpoint_file

    # Fallback: some released / renamed checkpoints may be a raw state_dict without the wrapper
    if "dims" not in checkpoint:
        # Attempt to infer variant name from filename
        variant = None
        if isinstance(name, str):
            fname = os.path.basename(name)
            m = re.match(r"OLMoASR-(tiny|base|small|medium|large)(?:-v2)?\.pt$", fname)
            if m:
                variant = m.group(1)
        if variant is None and name in MODEL2LINK:
            variant = name
        if variant is None:
            raise KeyError(
                "Checkpoint缺少'dims'字段且无法从文件名推断模型规格; 请使用官方名称下载, 或手动包装: {'dims': VARIANT_TO_DIMS[variant].__dict__, 'model_state_dict': state_dict}."
            )
        from olmoasr.config.model_dims import VARIANT_TO_DIMS, ModelDimensions as MDCls

        base_dims = VARIANT_TO_DIMS[variant]
        raw_state_dict = checkpoint  # entire mapping is parameters

        # 动态推断词表大小，以避免 size mismatch (常见 +1 情况)
        vocab_from_ckpt = None
        token_embed_key = "decoder.token_embedding.weight"
        if token_embed_key in raw_state_dict and hasattr(raw_state_dict[token_embed_key], "shape"):
            vocab_from_ckpt = raw_state_dict[token_embed_key].shape[0]

        if vocab_from_ckpt is not None and vocab_from_ckpt != base_dims.n_vocab:
            # 特殊处理: 一些英文模型导出的 raw state_dict 多出 1 行 (51865), 导致被判定为多语并解码乱码。
            if base_dims.n_vocab == 51864 and vocab_from_ckpt == 51865:
                # 裁掉最后一行保持英文单语行为
                token_embed_key = "decoder.token_embedding.weight"
                try:
                    raw_state_dict[token_embed_key] = raw_state_dict[token_embed_key][:51864, :]
                    print("[INFO] 检测到英文模型多出的 +1 词表行，已裁剪为 51864 以保持 English-only 解码。")
                    vocab_from_ckpt = 51864
                except Exception as e:
                    print(f"[WARN] 裁剪额外 vocab 行失败: {e}, 仍按 {vocab_from_ckpt} 加载 (可能导致多语模式乱码)")
            print(
                f"[INFO] 采用词表大小 {vocab_from_ckpt} (原预设 {base_dims.n_vocab}) 重建 dims"
            )
            dims = MDCls(
                n_mels=base_dims.n_mels,
                n_audio_ctx=base_dims.n_audio_ctx,
                n_audio_state=base_dims.n_audio_state,
                n_audio_head=base_dims.n_audio_head,
                n_audio_layer=base_dims.n_audio_layer,
                n_vocab=vocab_from_ckpt,
                n_text_ctx=base_dims.n_text_ctx,
                n_text_state=base_dims.n_text_state,
                n_text_head=base_dims.n_text_head,
                n_text_layer=base_dims.n_text_layer,
            )
        else:
            dims = base_dims

        if inference:
            model_instance = inf_model.OLMoASR(dims)
        else:
            model_instance = model.OLMoASR(dims)

        # 加载参数（现在维度应一致）
        load_res = model_instance.load_state_dict(raw_state_dict, strict=False)
        if isinstance(load_res, tuple):  # torch < 2.2 returns (missing, unexpected)
            missing, unexpected = load_res
            if missing:
                print(f"[WARN] 未匹配参数 (missing): {missing[:8]}{'...' if len(missing)>8 else ''}")
            if unexpected:
                print(f"[WARN] 额外参数 (unexpected): {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
        return model_instance.to(device)

    # Standard wrapped checkpoint path
    dims = ModelDimensions(**checkpoint["dims"])
    if inference:
        model_instance = inf_model.OLMoASR(dims)
    else:
        model_instance = model.OLMoASR(dims)
    model_instance.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model_instance.set_alignment_heads(alignment_heads)

    return model_instance.to(device)
