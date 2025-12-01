"""Standalone CoT-mode inference entrypoint inspired by inference/infer.py."""
import argparse
import copy
import os
import random
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from einops import rearrange
import gradio as gr
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parent
INFERENCE_DIR = ROOT / "inference"
XCODEC_DIR = INFERENCE_DIR / "xcodec_mini_infer"
PROMPT_DIR = ROOT / "prompt_egs"

# Make inference helpers importable without touching inference/infer.py
for extra_path in [INFERENCE_DIR, XCODEC_DIR, XCODEC_DIR / "descriptaudiocodec"]:
    sys.path.append(str(extra_path))

from codecmanipulator import CodecManipulator  # noqa: E402
from mmtokenizer import _MMSentencePieceTokenizer  # noqa: E402
from post_process_audio import replace_low_freq_with_energy_matched  # noqa: E402
from vocoder import build_codec_model, process_audio  # noqa: E402
from models.soundstream_hubert_new import SoundStream  # noqa: E402


def parse_args():
    default_stage1 = ROOT / "checkpoints" / "YuE-s1-7B-anneal-en-cot"
    default_stage2 = ROOT / "checkpoints" / "YuE-s2-1B-general"
    parser = argparse.ArgumentParser(description="CoT mode inference pipeline")
    parser.add_argument("--stage1_model", type=str, default=str(default_stage1))
    parser.add_argument("--stage2_model", type=str, default=str(default_stage2))
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--run_n_segments", type=int, default=2)
    parser.add_argument("--stage2_batch_size", type=int, default=4)
    parser.add_argument("--genre_txt", type=str, default=str(PROMPT_DIR / "genre.txt"))
    parser.add_argument("--lyrics_txt", type=str, default=str(PROMPT_DIR / "lyrics.txt"))
    parser.add_argument("--use_audio_prompt", action="store_true")
    parser.add_argument("--audio_prompt_path", type=str, default="")
    parser.add_argument("--prompt_start_time", type=float, default=0.0)
    parser.add_argument("--prompt_end_time", type=float, default=30.0)
    parser.add_argument("--use_dual_tracks_prompt", action="store_true")
    parser.add_argument("--vocal_track_prompt_path", type=str, default="")
    parser.add_argument("--instrumental_track_prompt_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "output"))
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--disable_offload_model", action="store_true")
    parser.add_argument("--cuda_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--basic_model_config",
        type=str,
        default=str(XCODEC_DIR / "final_ckpt" / "config.yaml"),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=str(XCODEC_DIR / "final_ckpt" / "ckpt_00360000.pth"),
    )
    parser.add_argument("--config_path", type=str, default=str(XCODEC_DIR / "decoders" / "config.yaml"))
    parser.add_argument(
        "--vocal_decoder_path",
        type=str,
        default=str(XCODEC_DIR / "decoders" / "decoder_131000.pth"),
    )
    parser.add_argument(
        "--inst_decoder_path",
        type=str,
        default=str(XCODEC_DIR / "decoders" / "decoder_151000.pth"),
    )
    parser.add_argument("-r", "--rescale", action="store_true")
    parser.add_argument("--cli", action="store_true", help="Run the pipeline via CLI only (no web UI).")
    parser.add_argument("--share", action="store_true", help="Share Gradio UI publicly.")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Gradio server host.")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port.")
    return parser.parse_args()


def print_run_config(args):
    summary = {
        "stage1_model": args.stage1_model,
        "stage2_model": args.stage2_model,
        "genre_txt": args.genre_txt,
        "lyrics_txt": args.lyrics_txt,
        "run_n_segments": args.run_n_segments,
        "stage2_batch_size": args.stage2_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "output_dir": args.output_dir,
        "cuda_idx": args.cuda_idx,
    }
    key_width = max(len(k) for k in summary) + 2
    print("\n=== YuE CoT Inference Configuration ===")
    for key, value in summary.items():
        print(f"{key.ljust(key_width)}: {value}")
    print("=" * 42 + "\n")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id: int, end_id: int):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def load_audio_mono(filepath: str, sampling_rate: int = 16000):
    audio, sr = torchaudio.load(filepath)
    audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def encode_audio(codec_model, audio_prompt, device, target_bw: float = 0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt = audio_prompt.unsqueeze(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    return raw_codes.cpu().numpy().astype(np.int16)


def split_lyrics(lyrics: str):
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    return [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]


def stage2_generate(model, prompt, codectool_stage1, tokenizer, device, batch_size: int = 16):
    codec_ids = codectool_stage1.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool_stage1.offset_tok_ids(
        codec_ids,
        global_offset=codectool_stage1.global_offset,
        codebook_size=codectool_stage1.codebook_size,
        num_codebooks=codectool_stage1.num_codebooks,
    ).astype(np.int32)

    if batch_size > 1:
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * 300
            idx_end = (i + 1) * 300
            codec_list.append(codec_ids[:, idx_begin:idx_end])
        codec_ids = np.concatenate(codec_list, axis=0)
        prompt_ids = np.concatenate(
            [
                np.tile([tokenizer.soa, tokenizer.stage_1], (batch_size, 1)),
                codec_ids,
                np.tile([tokenizer.stage_2], (batch_size, 1)),
            ],
            axis=1,
        )
    else:
        prompt_ids = np.concatenate(
            [
                np.array([tokenizer.soa, tokenizer.stage_1]),
                codec_ids.flatten(),
                np.array([tokenizer.stage_2]),
            ]
        ).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]
    block_list = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 46358),
        BlockTokenRangeProcessor(53526, tokenizer.vocab_size),
    ])

    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx : frames_idx + 1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids
        with torch.no_grad():
            stage2_output = model.generate(
                input_ids=input_ids,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=tokenizer.eoa,
                pad_token_id=tokenizer.eoa,
                logits_processor=block_list,
            )
        assert (
            stage2_output.shape[1] - prompt_ids.shape[1] == 7
        ), f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
        prompt_ids = stage2_output

    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]
    return output


def stage2_inference(
    model,
    stage1_output_set,
    output_dir,
    batch_size,
    codectool_stage1,
    codectool_stage2,
    tokenizer,
    device,
):
    stage2_result = []
    for i, stage1_file in enumerate(stage1_output_set):
        output_filename = os.path.join(output_dir, os.path.basename(stage1_file))
        if os.path.exists(output_filename):
            print(f"{output_filename} stage2 has done.")
            continue
        prompt = np.load(stage1_file).astype(np.int32)
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6
        if num_batch <= batch_size:
            output = stage2_generate(
                model,
                prompt[:, : output_duration * 50],
                codectool_stage1,
                tokenizer,
                device,
                batch_size=num_batch or 1,
            )
        else:
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)
            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)
                current_batch_size = (
                    batch_size
                    if seg != num_segments - 1 or num_batch % batch_size == 0
                    else num_batch % batch_size
                )
                segment = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    codectool_stage1,
                    tokenizer,
                    device,
                    batch_size=current_batch_size,
                )
                segments.append(segment)
            output = np.concatenate(segments, axis=0)
        if output_duration * 50 != prompt.shape[-1]:
            ending = stage2_generate(
                model,
                prompt[:, output_duration * 50 :],
                codectool_stage1,
                tokenizer,
                device,
                batch_size=1,
            )
            output = np.concatenate([output, ending], axis=0)
        output = codectool_stage2.ids2npy(output)
        fixed_output = copy.deepcopy(output)
        for line_idx, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[line_idx, j] = most_frequent
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    return stage2_result


def save_audio(wav: torch.Tensor, path: str, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    os.makedirs(folder_path, exist_ok=True)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)


def run_pipeline(args, genre_override: str | None = None, lyrics_override: str | None = None):
    print_run_config(args)
    if args.use_audio_prompt and not args.audio_prompt_path:
        raise FileNotFoundError("Provide --audio_prompt_path when --use_audio_prompt is set.")
    if args.use_dual_tracks_prompt and (
        not args.vocal_track_prompt_path or not args.instrumental_track_prompt_path
    ):
        raise FileNotFoundError("Provide both vocal and instrumental tracks when using dual prompts.")

    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    mmtokenizer = _MMSentencePieceTokenizer(str(INFERENCE_DIR / "mm_tokenizer_v0.2_hf" / "tokenizer.model"))
    model = AutoModelForCausalLM.from_pretrained(
        args.stage1_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model)

    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", 0, 8)
    model_config = OmegaConf.load(args.basic_model_config)
    orig_cwd = os.getcwd()
    try:
        os.chdir(str(INFERENCE_DIR))
        codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
    finally:
        os.chdir(orig_cwd)
    parameter_dict = torch.load(args.resume_path, map_location="cpu", weights_only=False)
    codec_model.load_state_dict(parameter_dict["codec_model"])
    codec_model.eval()

    stage1_output_dir = os.path.join(args.output_dir, "stage1")
    stage2_output_dir = os.path.join(args.output_dir, "stage2")
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)

    if genre_override is not None:
        genres = genre_override.strip()
    else:
        genres = Path(args.genre_txt).read_text().strip()
    if not genres:
        raise ValueError("Genre text is empty. Please provide tags.")
    if lyrics_override is not None:
        lyrics = split_lyrics(lyrics_override)
    else:
        lyrics = split_lyrics(Path(args.lyrics_txt).read_text())
    if not lyrics:
        raise ValueError("Lyrics text is empty or missing segments.")

    full_lyrics = "\n".join(lyrics)
    prompt_texts = [
        "Generate music from the given lyrics segment by segment.\n"
        f"[Genre] {genres}\n{full_lyrics}"
    ] + lyrics

    random_id = uuid.uuid4()
    max_new_tokens = args.max_new_tokens
    run_n_segments = min(args.run_n_segments + 1, len(lyrics))
    top_p = 0.93
    temperature = 1.0
    repetition_penalty = args.repetition_penalty
    start_of_segment = mmtokenizer.tokenize("[start_of_segment]")
    end_of_segment = mmtokenizer.tokenize("[end_of_segment]")
    stage1_output_set = []
    raw_output = None

    for i, p in enumerate(prompt_texts[:run_n_segments]):
        section_text = p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        guidance_scale = 1.5 if i <= 1 else 1.2
        if i == 0:
            continue
        if i == 1:
            if args.use_dual_tracks_prompt or args.use_audio_prompt:
                if args.use_dual_tracks_prompt:
                    vocals_wave = load_audio_mono(args.vocal_track_prompt_path)
                    instrumental_wave = load_audio_mono(args.instrumental_track_prompt_path)
                    vocal_codes = encode_audio(codec_model, vocals_wave, device, target_bw=0.5)
                    inst_codes = encode_audio(codec_model, instrumental_wave, device, target_bw=0.5)
                    vocal_ids = codectool.npy2ids(vocal_codes[0])
                    inst_ids = codectool.npy2ids(inst_codes[0])
                    ids_segment_interleaved = rearrange(
                        [np.array(vocal_ids), np.array(inst_ids)], "b n -> (n b)"
                    )
                    audio_prompt_codec = ids_segment_interleaved[
                        int(args.prompt_start_time * 50 * 2) : int(args.prompt_end_time * 50 * 2)
                    ].tolist()
                else:
                    audio_prompt = load_audio_mono(args.audio_prompt_path)
                    raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
                    code_ids = codectool.npy2ids(raw_codes[0])
                    audio_prompt_codec = code_ids[
                        int(args.prompt_start_time * 50) : int(args.prompt_end_time * 50)
                    ]
                audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [
                    mmtokenizer.eoa
                ]
                sentence_ids = (
                    mmtokenizer.tokenize("[start_of_reference]")
                    + audio_prompt_codec_ids
                    + mmtokenizer.tokenize("[end_of_reference]")
                )
                head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
            else:
                head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [
                mmtokenizer.soa
            ] + codectool.sep_ids
        else:
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [
                mmtokenizer.soa
            ] + codectool.sep_ids
        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            print(
                f"Section {i}: output length {input_ids.shape[-1]} exceeding context {max_context}, trimming."
            )
            input_ids = input_ids[:, -max_context:]
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList(
                    [
                        BlockTokenRangeProcessor(0, 32002),
                        BlockTokenRangeProcessor(32016, 32016),
                    ]
                ),
                guidance_scale=guidance_scale,
            )
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1] :]], dim=1)
        else:
            raw_output = output_seq

    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx) != len(eoa_idx):
        raise ValueError(
            f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}"
        )

    vocals = []
    instrumentals = []
    range_begin = 1 if args.use_audio_prompt or args.use_dual_tracks_prompt else 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
        instrumentals.append(instrumentals_ids)
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)

    vocal_save_path = os.path.join(
        stage1_output_dir,
        f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace(
            ".",
            "@",
        )
        + ".npy",
    )
    inst_save_path = os.path.join(
        stage1_output_dir,
        f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace(
            ".",
            "@",
        )
        + ".npy",
    )
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)
    stage1_output_set.extend([vocal_save_path, inst_save_path])

    if not args.disable_offload_model:
        model.cpu()
        del model
        torch.cuda.empty_cache()

    print("Stage 2 inference...")
    model_stage2 = AutoModelForCausalLM.from_pretrained(
        args.stage2_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model_stage2.eval()
    if torch.__version__ >= "2.0.0":
        model_stage2 = torch.compile(model_stage2)

    stage2_result = stage2_inference(
        model_stage2,
        stage1_output_set,
        stage2_output_dir,
        args.stage2_batch_size,
        codectool,
        codectool_stage2,
        mmtokenizer,
        device,
    )
    print(stage2_result)
    print("Stage 2 DONE.\n")

    recons_output_dir = os.path.join(args.output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, "mix")
    os.makedirs(recons_mix_dir, exist_ok=True)
    tracks = []
    for npy in stage2_result:
        codec_result = np.load(npy)
        decodec_rlt = []
        with torch.no_grad():
            decoded_waveform = codec_model.decode(
                torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
                .unsqueeze(0)
                .permute(1, 0, 2)
                .to(device)
            )
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        decodec_rlt.append(torch.as_tensor(decoded_waveform))
        decodec_rlt = torch.cat(decodec_rlt, dim=-1)
        save_path = os.path.join(
            recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3"
        )
        tracks.append(save_path)
        save_audio(decodec_rlt, save_path, 16000)

    latest_recons_mix = None
    for inst_path in tracks:
        try:
            if (inst_path.endswith(".wav") or inst_path.endswith(".mp3")) and "_itrack" in inst_path:
                vocal_path = inst_path.replace("_itrack", "_vtrack")
                if not os.path.exists(vocal_path):
                    continue
                recons_mix = os.path.join(
                    recons_mix_dir,
                    os.path.basename(inst_path).replace("_itrack", "_mixed"),
                )
                vocal_stem, sr = sf.read(inst_path)
                instrumental_stem, _ = sf.read(vocal_path)
                mix_stem = (vocal_stem + instrumental_stem)
                sf.write(recons_mix, mix_stem, sr)
                latest_recons_mix = recons_mix
        except Exception as exc:
            print(exc)

    vocal_decoder, inst_decoder = build_codec_model(
        args.config_path, args.vocal_decoder_path, args.inst_decoder_path
    )
    vocoder_output_dir = os.path.join(args.output_dir, "vocoder")
    vocoder_stems_dir = os.path.join(vocoder_output_dir, "stems")
    vocoder_mix_dir = os.path.join(vocoder_output_dir, "mix")
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)
    instrumental_output = None
    vocal_output = None
    for npy in stage2_result:
        if "_itrack" in npy:
            instrumental_output = process_audio(
                npy,
                os.path.join(vocoder_stems_dir, "itrack.mp3"),
                args.rescale,
                args,
                inst_decoder,
                codec_model,
            )
        else:
            vocal_output = process_audio(
                npy,
                os.path.join(vocoder_stems_dir, "vtrack.mp3"),
                args.rescale,
                args,
                vocal_decoder,
                codec_model,
            )

    latest_vocoder_mix = None
    if instrumental_output is not None and vocal_output is not None:
        try:
            mix_output = instrumental_output + vocal_output
            mix_basename = (
                os.path.basename(latest_recons_mix)
                if latest_recons_mix
                else "vocoder_mix_output.wav"
            )
            latest_vocoder_mix = os.path.join(vocoder_mix_dir, mix_basename)
            save_audio(mix_output, latest_vocoder_mix, 44100, args.rescale)
            print(f"Created mix: {latest_vocoder_mix}")
        except RuntimeError as exc:
            print(exc)
            print(
                "Mix failed! inst: "
                f"{instrumental_output.shape if instrumental_output is not None else 'None'}, "
                f"vocal: {vocal_output.shape if vocal_output is not None else 'None'}"
            )

    final_audio = None
    if latest_recons_mix and latest_vocoder_mix:
        final_audio = os.path.join(args.output_dir, os.path.basename(latest_recons_mix))
        replace_low_freq_with_energy_matched(
            a_file=latest_recons_mix,
            b_file=latest_vocoder_mix,
            c_file=final_audio,
            cutoff_freq=5500.0,
        )

    return {
        "stage1": stage1_output_set,
        "stage2": stage2_result,
        "recons_tracks": tracks,
        "recons_mix": latest_recons_mix,
        "vocoder_mix": latest_vocoder_mix,
        "final_audio": final_audio,
    }


def launch_ui(args):
    def safe_read_text(path):
        try:
            return Path(path).read_text().strip()
        except FileNotFoundError:
            return ""

    default_genres = safe_read_text(args.genre_txt)
    default_lyrics = safe_read_text(args.lyrics_txt)

    example_two_genre = (
        "female Cantonese Melancholic Classical airy vocal Piano bright vocal Pop Nostalgic Violin"
    )
    example_two_lyrics = """[verse]
我也并非无畏惧 我也并非不知险
然而在深情相拥之际 哪怕艰难
崇山峻岭 为你也愿作坦途
爱你无需讲道理


[verse]
但愿用真心实意去陪伴你
一想到心中的你 从未有过的勇气
猛然充满渐乏的身体


[chorus]
旁人全都阻拦 连道理也难容
仍全心奔赴 痛也不会哼
似追逐那片虹 谁人怎样猛冲
都难比我为你这般疯


[chorus]
前路荆棘再多 无人可阻我步
看着是千难万险都敢赴
我不够成熟 仅存有这份顽固



[verse]
我也渴望被关怀
但甘心成为卫士来守护你
荣耀你不赐予我 依旧执意走下去
洒脱笑着为你抗风雨


[chorus]
他人始终质疑 连常理也不顾
仍全心付出 苦都不觉苦
仍执着一份情 谁人道我顽固
亦不如我为你那般笃



[chorus]
前方阻碍重重 无人可让我停
看着那艰难险阻都前行
我并非无情 仅存这腔赤诚



[chorus]
摔下去再站起 就像是小强兵
明明已受挫折 仍努力去前行


[chorus]
他人从不认可 连道理也不通
仍全心付出爱都不觉浓
仍执着一段情 谁人怎样冲动
亦不如我为你那般疯



[chorus]
沿途风雨再猛 无人可阻我行
望着是千难万险都敢迎
哪怕无陪伴 依旧存这份坚定
期待情的人 全部情得很坚定

[bridge]

[verse]
""".strip()

    example_three_genre = "hiphop synthesizer tough male rap street bass vocal piano"
    example_three_lyrics = """[intro]
哥俩妙 别整那虚一套
都是敞亮人 兄弟你别闹
哥俩妙 别整那虚一套
你心里面藏招 他兜里揣着炮


[chorus]
人称酷佬 酷佬 酷佬 酷佬
街头酷佬
想要roll roll roll roll
富贵财宝
人称酷佬 酷佬 酷佬 酷佬
街头酷佬
想要roll roll roll roll
富贵财宝


[chorus]
都唤他江湖酷佬 ei 江湖酷佬
他渴望荣华财宝 ei 荣华财宝
世界能够那般广 世界亦那般渺
哎哟 你也听闻他 ayo 如此凑巧


[verse]
大伙都在讲你很牛啦 你本事确实大啦
我已然被你折服啦 O~
你跟那国外嘻哈咖挺像啦
光彩那是相当亮啦 胜过头顶的月亮啦
他的电话声响起ring ring
几句话里几句bling bling
零用钱换了ring ring
说再会 他讲pingping wu

他特豪爽 从不拖拉 能露脸的事儿绝不犯傻
戴着帽子 让你心神不宁 他一现身就好似寒风刮来
他知晓努力就有回报 却不明晓宽以待人 也不在意
渴望迅速拼搏一番 把你当工具使唤 走着山路 un


[verse]
他方言说得溜
遇人就喊好兄弟
撸起胳膊 欲做那把头
可他却偏爱那点小便宜
使他名声扫地
把他当挚友 可他把哥们当小弟
对他讲真心话他不珍惜 情谊渐渐地被他遗弃


[verse]
各类事情都想干
这兄弟特别悍
心中偶像周润发
他要做个街头霸
敢冲敢闯 你的善意他当是懦弱 引发争执与怒火
一回回把他当友可这般下去会走散
说对了你别夸俺 For free （这一条不收费哦）


[chorus]
人称酷佬 cool佬 cool佬 cool佬
江湖酷佬
想要roll roll roll roll
富贵财宝
人称酷佬 cool佬 cool佬 cool佬
江湖酷佬
想要roll roll roll roll
富贵财宝
都叫他江湖酷佬 ei 江湖酷佬
他想要富贵财宝 ei 富贵财宝
世界能够那么广 世界没有那么窄
哎哟 你也听闻他 ayo 如此凑巧


[verse]
哥们口气放得狂 自称街头王中王
金表得有十来块 神采飞扬够闪亮
场面一定得撑住 潇洒自如压个场
出门开辆法拉利 誓要成为强中强


[verse]
他事先对你一顿呛 习惯事后再给你点希望
不知不觉信任快耗光 临时补救是他的惯样
他仍然能四处闯荡
可情谊却难以再往
忧愁显在他的面庞
深夜的巷子里灯光下像只孤独的野狼


[verse]

渐渐大伙不捧场 他不明所以不转向
他常言未来路还长 但当下他的路摇晃

他的心境渐失常 独自之时难安详
他怨友朋 太吝啬太薄凉 却未看清己模样

并无深仇大恨藏 可为何又遇这状况
一笑能否解怨怅 ask me MR 江湖郎
望着家中财与宝 他不知如何是好
时而欢乐时而忧 做人实在太难熬 哎 这江湖郎


[outro]
哥俩妙 别总瞎胡闹
都是精明人 兄弟你别闹
哥俩妙 别总瞎胡闹""".strip()

    def _generate(genres_text, lyrics_text, run_segments, stage2_bs, max_tokens, rep_penalty):
        ui_args = copy.deepcopy(args)
        ui_args.run_n_segments = int(run_segments)
        ui_args.stage2_batch_size = int(stage2_bs)
        ui_args.max_new_tokens = int(max_tokens)
        ui_args.repetition_penalty = float(rep_penalty)
        ui_args.cli = True
        result = run_pipeline(
            ui_args,
            genre_override=genres_text or default_genres,
            lyrics_override=lyrics_text or default_lyrics,
        )
        return result.get("final_audio")

    with gr.Blocks(title="YuE CoT Web UI") as demo:
        gr.Markdown(
            "## YuE CoT 推理界面\n填写或调整提示后点击 Generate 生成音乐，可直接在下方试听。"
        )
        with gr.Row():
            genre_box = gr.Textbox(
                label="Genre Tags",
                value=default_genres,
                lines=4,
                placeholder="genre, mood, vocal timbre...",
            )
        lyrics_box = gr.Textbox(
            label="Lyrics",
            value=default_lyrics,
            lines=12,
            placeholder="[Verse]\nYour lyrics here",
        )
        gr.Examples(
            label="Examples",
            examples=[
                [default_genres, default_lyrics],
                [example_two_genre, example_two_lyrics],
                [example_three_genre, example_three_lyrics],
            ],
            inputs=[genre_box, lyrics_box],
            examples_per_page=3,
        )
        with gr.Row():
            run_segments_slider = gr.Slider(
                minimum=1,
                maximum=6,
                step=1,
                label="Run Segments",
                value=args.run_n_segments,
            )
            stage2_slider = gr.Slider(
                minimum=1,
                maximum=8,
                step=1,
                label="Stage2 Batch Size",
                value=args.stage2_batch_size,
            )
        with gr.Row():
            max_tokens_slider = gr.Slider(
                minimum=512,
                maximum=4096,
                step=64,
                label="Max New Tokens",
                value=args.max_new_tokens,
            )
            rep_pen_slider = gr.Slider(
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                label="Repetition Penalty",
                value=args.repetition_penalty,
            )
        generate_btn = gr.Button("Generate", variant="primary")
        audio_out = gr.Audio(label="Final Mix Preview", type="filepath")

        generate_btn.click(
            fn=_generate,
            inputs=[
                genre_box,
                lyrics_box,
                run_segments_slider,
                stage2_slider,
                max_tokens_slider,
                rep_pen_slider,
            ],
            outputs=[audio_out],
        )

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=False,
    )


def run():
    args = parse_args()
    if args.cli:
        run_pipeline(args)
    else:
        launch_ui(args)


if __name__ == "__main__":
    run()
