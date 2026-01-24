#!/usr/bin/env python
"""
VibeVoice ASR Gradio 演示程序
基于 demo/vibevoice_asr_gradio_demo.py 创建的独立程序
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import time
import json
import gradio as gr
from typing import List, Dict, Tuple, Optional, Generator
import tempfile
import base64
import io
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import TextIteratorStreamer for streaming generation
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    # Only apply RoPE, RMSNorm, SwiGLU patches (these affect the underlying Qwen2 layers)
    apply_liger_kernel_to_qwen2(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
    )
    print("✅ Liger Kernel 已应用到 Qwen2 组件 (RoPE, RMSNorm, SwiGLU)")
except Exception as e:
    print(f"⚠️ 应用 Liger Kernel 失败: {e}, 可通过以下命令安装: pip install liger-kernel")
    
# Try to import pydub for MP3 conversion
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("⚠️ 警告: pydub 不可用，将使用 WAV 格式")

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg, COMMON_AUDIO_EXTS


# 默认模型路径（本地预下载的模型）
DEFAULT_MODEL_PATH = "checkpoints/VibeVoice-ASR"


class VibeVoiceASRInference:
    """VibeVoice ASR 模型推理封装类。"""
    
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "flash_attention_2"):
        """
        初始化 ASR 推理管道。
        
        Args:
            model_path: 预训练模型路径（HuggingFace 格式目录或模型名称）
            device: 运行推理的设备
            dtype: 模型权重数据类型
            attn_implementation: 注意力实现方式 ('flash_attention_2', 'sdpa', 'eager')
        """
        print(f"正在从 {model_path} 加载 VibeVoice ASR 模型")
        
        # Load processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        
        # Load model
        print(f"使用注意力实现: {attn_implementation}")
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device if device == "auto" else None,
            attn_implementation=attn_implementation,
            trust_remote_code=True
        )
        
        if device != "auto":
            self.model = self.model.to(device)
        
        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ 模型加载成功，运行于 {self.device}")
        print(f"📊 总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    
    def transcribe(
        self, 
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sample_rate: int = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        context_info: str = None,
        streamer: Optional[TextIteratorStreamer] = None,
    ) -> dict:
        """
        将音频转录为文本。
        
        Args:
            audio_path: 音频文件路径
            audio_array: 音频数组（如果不从文件加载）
            sample_rate: 音频数组的采样率
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（0 表示贪婪解码）
            top_p: 核采样的 Top-p 值（1.0 表示不过滤）
            do_sample: 是否使用采样
            num_beams: 束搜索的束数（1 表示贪婪解码）
            repetition_penalty: 重复惩罚（1.0 表示无惩罚）
            context_info: 可选的上下文信息（如热词、说话人名称、主题等）
            streamer: 可选的 TextIteratorStreamer 用于流式输出
            
        Returns:
            包含转录结果的字典
        """
        # Process audio
        inputs = self.processor(
            audio=audio_path,
            sampling_rate=sample_rate,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Generate
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # Add streamer if provided
        if streamer is not None:
            generation_config["streamer"] = streamer
        
        # Add stopping criteria for stop button support
        generation_config["stopping_criteria"] = StoppingCriteriaList([StopOnFlag()])
        
        # Remove None values
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        
        start_time = time.time()
        
        # Calculate input token statistics before generation
        input_ids = inputs['input_ids'][0]  # Shape: [seq_len]
        total_input_tokens = input_ids.shape[0]
        
        # Count padding tokens (tokens equal to pad_id)
        pad_id = self.processor.pad_id
        padding_mask = (input_ids == pad_id)
        num_padding_tokens = padding_mask.sum().item()
        
        # Count speech tokens (tokens between speech_start_id and speech_end_id)
        speech_start_id = self.processor.speech_start_id
        speech_end_id = self.processor.speech_end_id
        
        # Find speech regions
        input_ids_list = input_ids.tolist()
        num_speech_tokens = 0
        in_speech = False
        for token_id in input_ids_list:
            if token_id == speech_start_id:
                in_speech = True
                num_speech_tokens += 1  # Count speech_start token
            elif token_id == speech_end_id:
                in_speech = False
                num_speech_tokens += 1  # Count speech_end token
            elif in_speech:
                num_speech_tokens += 1
        
        # Text tokens = total - speech - padding
        num_text_tokens = total_input_tokens - num_speech_tokens - num_padding_tokens
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode output
        generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Parse structured output
        try:
            transcription_segments = self.processor.post_process_transcription(generated_text)
        except Exception as e:
            print(f"警告: 解析结构化输出失败: {e}")
            transcription_segments = []
        
        return {
            "raw_text": generated_text,
            "segments": transcription_segments,
            "generation_time": generation_time,
            "input_tokens": {
                "total": total_input_tokens,
                "speech": num_speech_tokens,
                "text": num_text_tokens,
                "padding": num_padding_tokens,
            },
        }


def clip_and_encode_audio(
    audio_data: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    segment_idx: int,
    use_mp3: bool = True,
    target_sr: int = 16000,
    mp3_bitrate: str = "32k"
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    裁剪音频片段并编码为 base64。
    
    Args:
        audio_data: 完整音频数组
        sr: 采样率
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        segment_idx: 片段索引
        use_mp3: 是否使用 MP3 格式（更小）
        target_sr: 目标采样率
        mp3_bitrate: MP3 比特率
        
    Returns:
        元组 (segment_idx, base64_string, error_message)
    """
    try:
        # Convert time to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return segment_idx, None, f"无效的时间范围: [{start_time:.2f}s - {end_time:.2f}s]"
        
        # Extract segment
        segment_data = audio_data[start_sample:end_sample]
        
        # Downsample if needed
        if sr != target_sr and target_sr < sr:
            duration = len(segment_data) / sr
            new_length = int(duration * target_sr)
            indices = np.linspace(0, len(segment_data) - 1, new_length)
            segment_data = np.interp(indices, np.arange(len(segment_data)), segment_data)
            sr = target_sr
        
        # Convert float32 audio to int16 for encoding
        segment_data_int16 = (segment_data * 32768.0).astype(np.int16)
        
        # Convert to MP3 if pydub is available and use_mp3 is True
        if use_mp3 and HAS_PYDUB:
            try:
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, segment_data_int16, sr, format='WAV', subtype='PCM_16')
                wav_buffer.seek(0)
                
                audio_segment = AudioSegment.from_wav(wav_buffer)
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                mp3_buffer = io.BytesIO()
                audio_segment.export(mp3_buffer, format='mp3', bitrate=mp3_bitrate)
                mp3_buffer.seek(0)
                
                audio_bytes = mp3_buffer.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_src = f"data:audio/mp3;base64,{audio_base64}"
                
                return segment_idx, audio_src, None
            except Exception as e:
                print(f"片段 {segment_idx} MP3 转换失败，使用 WAV: {e}")
        
        # Fall back to WAV format
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, segment_data_int16, sr, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        audio_bytes = wav_buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_src = f"data:audio/wav;base64,{audio_base64}"
        
        return segment_idx, audio_src, None
        
    except Exception as e:
        error_msg = f"裁剪片段 {segment_idx} 时出错: {str(e)}"
        print(error_msg)
        return segment_idx, None, error_msg


def extract_audio_segments(audio_path: str, segments: List[Dict]) -> List[Tuple[str, str, Optional[str]]]:
    """
    从音频文件中高效提取多个片段（使用并行处理）。
    
    Args:
        audio_path: 原始音频文件路径
        segments: 包含 start_time, end_time 等信息的片段字典列表
    
    Returns:
        元组列表 (segment_label, audio_base64_src, error_msg)
    """
    try:
        print(f"📂 正在加载音频文件: {audio_path}")
        audio_data, sr = load_audio_use_ffmpeg(audio_path, resample=False)
        print(f"✅ 音频加载完成: {len(audio_data)} 采样点, {sr} Hz")
        
        tasks = []
        use_mp3 = HAS_PYDUB
        
        for i, seg in enumerate(segments):
            start_time = seg.get('start_time')
            end_time = seg.get('end_time')
            
            if (not isinstance(start_time, (int, float)) or 
                not isinstance(end_time, (int, float)) or 
                start_time >= end_time):
                tasks.append((i, None, None, None, None, None))
                continue
            
            tasks.append((audio_data, sr, start_time, end_time, i, use_mp3))
        
        results = []
        total_segments = len(tasks)
        completed_count = 0
        
        max_workers = os.cpu_count() or 4
        print(f"🚀 使用 {max_workers} 个线程并行处理...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for task in tasks:
                if task[0] is None:
                    continue
                future = executor.submit(clip_and_encode_audio, *task)
                futures[future] = task[4]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    if completed_count % 100 == 0 or completed_count == len(futures):
                        print(f"进度: {completed_count}/{len(futures)} 个片段已处理 ({completed_count*100//len(futures)}%)")
                except Exception as e:
                    idx = futures[future]
                    results.append((idx, None, f"处理错误: {str(e)}"))
                    completed_count += 1
                    print(f"片段 {idx} 处理错误: {e}")
        
        print(f"✅ 完成处理所有 {len(futures)} 个有效片段")
        
        results.sort(key=lambda x: x[0])
        
        audio_segments = []
        for i, (idx, audio_src, error_msg) in enumerate(results):
            seg = segments[idx] if idx < len(segments) else {}
            start_time = seg.get('start_time', 'N/A')
            end_time = seg.get('end_time', 'N/A')
            speaker_id = seg.get('speaker_id', 'N/A')
            
            segment_label = f"片段 {idx+1}: [{start_time:.2f}s - {end_time:.2f}s] 说话人 {speaker_id}"
            audio_segments.append((segment_label, audio_src, error_msg))
        
        return audio_segments
        
    except Exception as e:
        print(f"加载音频文件时出错: {e}")
        return []


# Global variable to store the ASR model
asr_model = None

# Global stop flag for generation
stop_generation_flag = False


class StopOnFlag(StoppingCriteria):
    """自定义停止条件，检查全局标志。"""
    def __call__(self, input_ids, scores, **kwargs):
        global stop_generation_flag
        return stop_generation_flag


def parse_time_to_seconds(val: Optional[str]) -> Optional[float]:
    """将秒数或 hh:mm:ss 格式解析为浮点秒数。"""
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        pass
    if ":" in val:
        parts = val.split(":")
        if not all(p.strip().replace(".", "", 1).isdigit() for p in parts):
            return None
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            return None
        return h * 3600 + m * 60 + s
    return None


def slice_audio_to_temp(
    audio_data: np.ndarray,
    sample_rate: int,
    start_sec: Optional[float],
    end_sec: Optional[float]
) -> Tuple[Optional[str], Optional[str]]:
    """将 audio_data 裁剪到 [start_sec, end_sec) 并写入临时 WAV 文件。"""
    n_samples = len(audio_data)
    full_duration = n_samples / float(sample_rate)
    start = 0.0 if start_sec is None else max(0.0, start_sec)
    end = full_duration if end_sec is None else min(full_duration, end_sec)
    if end <= start:
        return None, f"无效的时间范围: start={start:.2f}s, end={end:.2f}s"
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment = audio_data[start_idx:end_idx]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.close()
    segment_int16 = (segment * 32768.0).astype(np.int16)
    sf.write(temp_file.name, segment_int16, sample_rate, subtype='PCM_16')
    return temp_file.name, None


def initialize_model(model_path: str, device: str = "cuda", attn_implementation: str = "flash_attention_2"):
    """初始化 ASR 模型。"""
    global asr_model
    try:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        asr_model = VibeVoiceASRInference(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation
        )
        return f"✅ 模型从 {model_path} 加载成功"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ 加载模型时出错: {str(e)}"


def transcribe_audio(
    audio_input,
    audio_path_input: str,
    start_time_input: str,
    end_time_input: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float = 1.0,
    context_info: str = ""
) -> Generator[Tuple[str, str], None, None]:
    """
    转录音频并返回带有音频片段的结果（流式版本）。
    
    Args:
        audio_input: 音频文件路径或元组 (sample_rate, audio_data)
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度（0 表示贪婪解码）
        top_p: 核采样的 Top-p 值
        do_sample: 是否使用采样
        context_info: 可选的上下文信息
    
    Yields:
        元组 (raw_text, audio_segments_html)
    """
    if asr_model is None:
        yield "❌ 请先加载模型！", ""
        return
    
    if not audio_path_input and audio_input is None:
        yield "❌ 请提供音频输入！", ""
        return
    
    try:
        print("[信息] 收到转录请求")
        start_sec = parse_time_to_seconds(start_time_input)
        end_sec = parse_time_to_seconds(end_time_input)
        print(f"[信息] 解析的时间范围: start={start_sec}, end={end_sec}")
        if (start_time_input and start_sec is None) or (end_time_input and end_sec is None):
            yield "❌ 无效的时间格式。请使用秒数或 hh:mm:ss 格式。", ""
            return

        audio_path = None
        audio_array = None
        sample_rate = None

        if audio_path_input:
            candidate = Path(audio_path_input.strip())
            if not candidate.exists():
                yield f"❌ 指定的路径不存在: {candidate}", ""
                return
            audio_path = str(candidate)
            print(f"[信息] 使用指定的音频路径: {audio_path}")
        elif isinstance(audio_input, str):
            audio_path = audio_input
            print(f"[信息] 使用上传的音频路径: {audio_path}")
        elif isinstance(audio_input, tuple):
            sample_rate, audio_array = audio_input
            print(f"[信息] 收到麦克风音频，采样率={sample_rate}")
        elif audio_path is None:
            yield "❌ 无效的音频输入格式！", ""
            return

        # If slicing is requested, load and slice audio
        if start_sec is not None or end_sec is not None:
            print("[信息] 按请求的时间范围裁剪音频")
            if audio_array is None or sample_rate is None:
                try:
                    audio_array, sample_rate = load_audio_use_ffmpeg(audio_path, resample=False)
                    print("[信息] 通过 ffmpeg 加载音频用于裁剪")
                except Exception as exc:
                    yield f"❌ 加载音频进行裁剪失败: {exc}", ""
                    return
            sliced_path, err = slice_audio_to_temp(audio_array, sample_rate, start_sec, end_sec)
            if err:
                yield f"❌ {err}", ""
                return
            audio_path = sliced_path
            print(f"[信息] 裁剪后的音频已写入临时文件: {audio_path}")
        elif audio_array is not None and sample_rate is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_path = temp_file.name
            temp_file.close()
            audio_data_int16 = (audio_array * 32768.0).astype(np.int16)
            sf.write(audio_path, audio_data_int16, sample_rate, subtype='PCM_16')
            print(f"[信息] 麦克风音频已保存到临时文件: {audio_path}")
        
        # Create streamer for real-time output
        streamer = TextIteratorStreamer(
            asr_model.processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        result_container = {"result": None, "error": None}
        
        def run_transcription():
            try:
                result_container["result"] = asr_model.transcribe(
                    audio_path=audio_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    context_info=context_info if context_info and context_info.strip() else None,
                    streamer=streamer
                )
            except Exception as e:
                result_container["error"] = str(e)
                traceback.print_exc()
        
        print("[信息] 开始模型转录（流式模式）")
        start_time = time.time()
        transcription_thread = threading.Thread(target=run_transcription)
        transcription_thread.start()
        
        # Yield streaming output
        generated_text = ""
        token_count = 0
        for new_text in streamer:
            generated_text += new_text
            token_count += 1
            elapsed = time.time() - start_time
            formatted_text = generated_text.replace('},', '},\n')
            streaming_output = f"--- 🔴 实时流式输出 (tokens: {token_count}, 时间: {elapsed:.1f}s) ---\n{formatted_text}"
            yield streaming_output, "<div style='padding: 20px; text-align: center; color: #6c757d;'>⏳ 正在生成转录结果... 完成后将显示音频片段。</div>"
        
        transcription_thread.join()
        
        if result_container["error"]:
            yield f"❌ 转录过程中出错: {result_container['error']}", ""
            return
        
        result = result_container["result"]
        generation_time = time.time() - start_time
        
        input_tokens = result.get('input_tokens', {})
        speech_tokens = input_tokens.get('speech', 0)
        text_tokens = input_tokens.get('text', 0)
        padding_tokens = input_tokens.get('padding', 0)
        total_input = input_tokens.get('total', 0)
        
        raw_output = f"--- ✅ 原始输出 ---\n"
        raw_output += f"📥 输入: {total_input} tokens (🎤 语音: {speech_tokens}, 📝 文本: {text_tokens}, ⬜ 填充: {padding_tokens})\n"
        raw_output += f"📤 输出: {token_count} tokens | ⏱️ 时间: {generation_time:.2f}s\n"
        raw_output += f"---\n"
        formatted_raw_text = result['raw_text'].replace('},', '},\n')
        raw_output += formatted_raw_text
        
        print(f"[调试] 原始模型输出:")
        print(f"[调试] {result['raw_text']}")
        print(f"[调试] 找到 {len(result['segments'])} 个片段")
        
        audio_segments_html = ""
        segments = result['segments']
        
        if segments:
            num_segments = len(segments)
            print(f"[信息] 创建每个片段的音频剪辑 ({num_segments} 个片段, 16kHz mono MP3 @ 32kbps)")
            
            audio_segments = extract_audio_segments(audio_path, segments)
            print("[信息] 完成创建音频剪辑")
            
            total_duration = sum(
                (seg.get('end_time', 0) - seg.get('start_time', 0)) 
                for seg in segments 
                if isinstance(seg.get('start_time'), (int, float)) and isinstance(seg.get('end_time'), (int, float))
            )
            approx_size_kb = total_duration * 4
            
            theme_css = """
            <style>
            :root {
                --segment-bg: #f8f9fa;
                --segment-border: #e1e5e9;
                --segment-text: #495057;
                --segment-meta: #6c757d;
                --content-bg: white;
                --content-border: #007bff;
                --warning-bg: #fff3cd;
                --warning-border: #ffc107;
                --warning-text: #856404;
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --segment-bg: #2d3748;
                    --segment-border: #4a5568;
                    --segment-text: #e2e8f0;
                    --segment-meta: #a0aec0;
                    --content-bg: #1a202c;
                    --content-border: #4299e1;
                    --warning-bg: #744210;
                    --warning-border: #d69e2e;
                    --warning-text: #faf089;
                }
            }
            
            .audio-segments-container {
                max-height: 600px;
                overflow-y: auto;
                padding: 10px;
            }
            
            .audio-segment {
                margin-bottom: 15px;
                padding: 15px;
                border: 2px solid var(--segment-border);
                border-radius: 8px;
                background-color: var(--segment-bg);
                transition: all 0.3s ease;
            }
            
            .audio-segment:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            
            .segment-header {
                margin-bottom: 10px;
            }
            
            .segment-title {
                margin: 0;
                color: var(--segment-text);
                font-size: 16px;
                font-weight: 600;
            }
            
            .segment-meta {
                margin-top: 5px;
                font-size: 14px;
                color: var(--segment-meta);
            }
            
            .segment-content {
                margin-bottom: 10px;
                padding: 12px;
                background-color: var(--content-bg);
                border-radius: 6px;
                border-left: 4px solid var(--content-border);
                color: var(--segment-text);
                line-height: 1.5;
            }
            
            .segment-audio {
                width: 100%;
                margin-top: 10px;
                border-radius: 4px;
            }
            
            .segment-warning {
                margin-top: 10px;
                padding: 10px;
                background-color: var(--warning-bg);
                border-radius: 4px;
                border-left: 4px solid var(--warning-border);
                color: var(--warning-text);
                font-size: 13px;
            }
            
            .segments-title {
                color: var(--segment-text);
                margin-bottom: 10px;
            }
            
            .segments-description {
                color: var(--segment-meta);
                margin-bottom: 20px;
            }
            
            .size-badge {
                display: inline-block;
                background: linear-gradient(135deg, #6c757d, #495057);
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }
            </style>
            """
            
            audio_segments_html = theme_css
            audio_segments_html += f"<div class='audio-segments-container'>"
            
            format_info = "MP3 32kbps 16kHz mono" if HAS_PYDUB else "WAV 16kHz"
            audio_segments_html += f"<h3 class='segments-title'>🔊 音频片段 ({num_segments} 个片段)"
            audio_segments_html += f"<span class='size-badge'>📦 ~{approx_size_kb:.0f}KB ({format_info})</span></h3>"
            audio_segments_html += "<p class='segments-description'>🎵 点击播放按钮可直接收听每个片段！</p>"
            
            for i, (label, audio_src, error_msg) in enumerate(audio_segments):
                seg = segments[i] if i < len(segments) else {}
                start_time = seg.get('start_time', 'N/A')
                end_time = seg.get('end_time', 'N/A')
                speaker_id = seg.get('speaker_id', 'N/A')
                content = seg.get('text', '')
                
                start_str = f"{start_time:.2f}" if isinstance(start_time, (int, float)) else str(start_time)
                end_str = f"{end_time:.2f}" if isinstance(end_time, (int, float)) else str(end_time)
                
                audio_segments_html += f"""
                <div class='audio-segment'>
                    <div class='segment-header'>
                        <h4 class='segment-title'>片段 {i+1}</h4>
                        <div class='segment-meta'>
                            <strong>时间:</strong> [{start_str}s - {end_str}s] | 
                            <strong>说话人:</strong> {speaker_id}
                        </div>
                    </div>
                    
                    <div class='segment-content'>
                        {content}
                    </div>
                """
                
                if audio_src:
                    audio_type = 'audio/mp3' if 'audio/mp3' in audio_src else 'audio/wav'
                    audio_segments_html += f"""
                    <audio controls class='segment-audio' preload='none'>
                        <source src='{audio_src}' type='{audio_type}'>
                        您的浏览器不支持音频播放。
                    </audio>
                    """
                elif error_msg:
                    audio_segments_html += f"""
                    <div class='segment-warning'>
                        <small>❌ {error_msg}</small>
                    </div>
                    """
                else:
                    audio_segments_html += """
                    <div class='segment-warning'>
                        <small>此片段无法播放音频</small>
                    </div>
                    """
                
                audio_segments_html += "</div>"
            
            audio_segments_html += "</div>"
        else:
            audio_segments_html = """
            <style>
            :root {
                --no-segments-text: #6c757d;
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --no-segments-text: #a0aec0;
                }
            }
            
            .no-segments-container {
                padding: 20px;
                text-align: center;
                color: var(--no-segments-text);
                line-height: 1.6;
            }
            </style>
            <div class='no-segments-container'>
                <p>❌ 没有可用的音频片段。</p>
                <p>这可能是因为模型输出中不包含有效的时间戳。</p>
            </div>
            """
        
        yield raw_output, audio_segments_html
        
    except Exception as e:
        print(f"转录过程中出错: {e}")
        print(traceback.format_exc())
        yield f"❌ 转录过程中出错: {str(e)}", ""


def create_gradio_interface(model_path: str, default_max_tokens: int = 8192, attn_implementation: str = "flash_attention_2"):
    """创建并启动 Gradio 界面。
    
    Args:
        model_path: 模型路径（HuggingFace 格式目录或模型名称）
        default_max_tokens: max_new_tokens 滑块的默认值
        attn_implementation: 注意力实现方式
    """
    
    # Initialize model at startup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_status = initialize_model(model_path, device, attn_implementation)
    print(model_status)
    
    if model_status.startswith("❌"):
        print("\n" + "="*80)
        print("💥 致命错误: 模型加载失败！")
        print("="*80)
        print("无法启动演示，没有有效的模型。请检查:")
        print("  1. 模型路径是否正确")
        print("  2. 模型文件是否损坏")
        print("  3. 是否有足够的 GPU 内存")
        print("  4. CUDA 是否正确安装（如果使用 GPU）")
        print("="*80)
        sys.exit(1)
    
    # Custom CSS for Stop button styling
    custom_css = """
    #stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: none !important;
        color: white !important;
    }
    #stop-btn:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    }
    .youtube-banner {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .youtube-banner a {
        color: white;
        text-decoration: none;
        font-weight: bold;
    }
    .youtube-banner a:hover {
        text-decoration: underline;
    }
    """
    
    with gr.Blocks(title="VibeVoice ASR 语音识别演示") as demo:
        # YouTube 频道信息
        gr.HTML("""
        <div class="youtube-banner">
            <span style="font-size: 24px;">📺</span>
            <span style="font-size: 18px; margin-left: 10px;">欢迎访问我的 YouTube 频道：</span>
            <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="font-size: 20px; margin-left: 10px;">
                🎬 AI 技术分享频道
            </a>
            <span style="font-size: 14px; margin-left: 15px; opacity: 0.9;">| 订阅获取更多 AI 技术教程</span>
        </div>
        """)
        
        gr.Markdown("# 🎙️ VibeVoice ASR 语音识别演示")
        gr.Markdown("上传音频文件或通过麦克风录音，即可获得带有说话人分离的语音转文字结果。")
        gr.Markdown(f"**已加载模型:** `{model_path}`")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Generation parameters
                gr.Markdown("## ⚙️ 生成参数")
                max_tokens_slider = gr.Slider(
                    minimum=4096,
                    maximum=65536,
                    value=default_max_tokens,
                    step=4096,
                    label="最大生成 Token 数"
                )
                
                # Sampling parameters
                gr.Markdown("### 🎲 采样设置")
                do_sample_checkbox = gr.Checkbox(
                    value=False,
                    label="启用采样",
                    info="启用随机采样而非确定性解码"
                )
                
                with gr.Column(visible=False) as sampling_params:
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        label="温度 (Temperature)",
                        info="0 = 贪婪解码，越高越随机"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Top-p (核采样)",
                        info="1.0 = 不过滤"
                    )
                
                repetition_penalty_slider = gr.Slider(
                    minimum=1.0,
                    maximum=1.2,
                    value=1.0,
                    step=0.01,
                    label="重复惩罚",
                    info="1.0 = 无惩罚，越高重复越少（适用于贪婪解码和采样）"
                )
                
                # Context information section
                gr.Markdown("## 📋 上下文信息（可选）")
                context_info_input = gr.Textbox(
                    label="上下文信息",
                    placeholder="Enter hotwords, speaker names, topics, or other context to help transcription...\nExample:\nJohn Smith\nMachine Learning\nOpenAI",
                    lines=4,
                    max_lines=8,
                    interactive=True,
                    info="提供热词、专业术语或说话人名称等上下文信息以提高准确性"
                )
            
            with gr.Column(scale=2):
                # Audio input section
                gr.Markdown("## 🎵 音频输入")
                audio_input = gr.Audio(
                    label="上传音频文件或通过麦克风录音",
                    sources=["upload", "microphone"],
                    type="filepath",
                    interactive=True,
                    buttons=["download"]
                )
                
                with gr.Accordion("📂 高级选项：远程路径 & 时间裁剪", open=False):
                    audio_path_input = gr.Textbox(
                        label="音频路径（可选）",
                        placeholder="输入远程音频文件路径",
                        lines=1
                    )
                    with gr.Row():
                        start_time_input = gr.Textbox(
                            label="开始时间",
                            placeholder="例如: 0 或 00:00:00",
                            lines=1,
                            info="留空则从开头开始"
                        )
                        end_time_input = gr.Textbox(
                            label="结束时间",
                            placeholder="例如: 30.5 或 00:00:30.5",
                            lines=1,
                            info="留空则使用完整长度"
                        )
                
                with gr.Row():
                    transcribe_button = gr.Button("🎯 开始转录", variant="primary", size="lg", scale=3)
                    stop_button = gr.Button("⏹️ 停止", variant="secondary", size="lg", scale=1, elem_id="stop-btn")
                
                # Results section
                gr.Markdown("## 📝 转录结果")
                
                with gr.Tabs():
                    with gr.TabItem("原始输出"):
                        raw_output = gr.Textbox(
                            label="原始转录输出",
                            lines=8,
                            max_lines=20,
                            interactive=False
                        )
                    
                    with gr.TabItem("音频片段"):
                        audio_segments_output = gr.HTML(
                            label="播放各个片段以验证准确性"
                        )
        
        # Event handlers
        do_sample_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[do_sample_checkbox],
            outputs=[sampling_params]
        )
        
        def reset_stop_flag():
            """重置停止标志。"""
            global stop_generation_flag
            stop_generation_flag = False
        
        def set_stop_flag():
            """设置停止标志以中断生成。"""
            global stop_generation_flag
            stop_generation_flag = True
            return "⏹️ 已请求停止..."
        
        transcribe_button.click(
            fn=reset_stop_flag,
            inputs=[],
            outputs=[],
            queue=False
        ).then(
            fn=transcribe_audio,
            inputs=[
                audio_input,
                audio_path_input,
                start_time_input,
                end_time_input,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                do_sample_checkbox,
                repetition_penalty_slider,
                context_info_input
            ],
            outputs=[raw_output, audio_segments_output]
        )
        
        stop_button.click(
            fn=set_stop_flag,
            inputs=[],
            outputs=[raw_output],
            queue=False
        )
        
        # Add examples
        gr.Markdown("## 📋 使用说明")
        gr.Markdown(f"""
        1. **上传音频**: 使用音频组件上传文件或通过麦克风录音
           - **支持的格式**: {', '.join(sorted(set([ext.lower() for ext in COMMON_AUDIO_EXTS])))}
           - 可选：设置**开始/结束时间**（秒数或 hh:mm:ss 格式）以在转录前裁剪音频
        2. **上下文信息（可选）**: 提供上下文以提高转录准确性
           - 添加热词、专有名词、说话人姓名或技术术语
           - 每行一项或逗号分隔
           - 示例: "John Smith", "OpenAI", "machine learning"
        3. **调整参数**: 根据需要配置生成参数
        4. **开始转录**: 点击"开始转录"按钮获取结果
        5. **查看结果**: 
           - **原始输出**: 查看模型的原始输出
           - **音频片段**: 直接播放各个片段以验证准确性
        
        **音频片段**: 每个片段显示时间范围、说话人 ID、转录内容，以及可直接播放验证的嵌入式音频播放器。
        """)
    
    return demo, custom_css


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Gradio 演示")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=DEFAULT_MODEL_PATH,
        help="模型路径（HuggingFace 格式目录或模型名称）"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="注意力实现方式（默认: flash_attention_2）"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="默认最大生成 token 数（默认: 32768）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器绑定的主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务器绑定的端口"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公共链接"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    demo, custom_css = create_gradio_interface(
        model_path=args.model_path,
        default_max_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation
    )
    
    print(f"🚀 正在启动 VibeVoice ASR 演示...")
    print(f"📍 服务器地址: http://{args.host}:{args.port}")
    
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "show_error": True,
        "theme": gr.themes.Soft(),
        "css": custom_css,
    }
    
    demo.queue(default_concurrency_limit=3)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
