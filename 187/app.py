"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import os
import traceback

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        """Initialize without immediately loading model (lazy load)."""
        self.model_path = model_path  # initial preferred path (may change via UI)
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        self.model = None
        self.processor = None
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts

    # ---------------------- Model & Voices ----------------------
    def load_model(self):
        """Load the VibeVoice model and processor (called lazily)."""
        print(f"Loading processor & model from {self.model_path}")
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        # Load model
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map='cuda' if torch.cuda.is_available() else None,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        # Configure scheduler & steps
        try:
            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                self.model.model.noise_scheduler.config,
                algorithm_type='sde-dpmsolver++',
                beta_schedule='squaredcos_cap_v2'
            )
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        except Exception as e:
            print(f"Warning configuring scheduler: {e}")
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")

    def setup_voice_presets(self):
        """Setup voice presets by scanning voices directories (supports both demo/voices and voices)."""
        root_dir = os.path.dirname(__file__)
        candidate_dirs = [
            os.path.join(root_dir, 'demo', 'voices'),
            os.path.join(root_dir, 'voices'),
        ]
        voices_dir = None
        for d in candidate_dirs:
            if os.path.exists(d):
                voices_dir = d
                break
        if voices_dir is None:
            print("Warning: Voices directory not found in expected locations.")
            self.voice_presets = {}
            self.available_voices = {}
            return
        wav_files = [f for f in os.listdir(voices_dir)
                     if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and
                     os.path.isfile(os.path.join(voices_dir, f))]
        self.voice_presets = {}
        for wav in wav_files:
            name = os.path.splitext(wav)[0]
            self.voice_presets[name] = os.path.join(voices_dir, wav)
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {n: p for n, p in self.voice_presets.items() if os.path.exists(p)}
        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .wav files to demo/voices directory.")
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and (if needed) resample an audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def ensure_model_loaded(self, selected_model_path: str):
        if self.model is not None and self.model_path == selected_model_path:
            return
        if self.model is not None and self.model_path != selected_model_path:
            try:
                print(f"Unloading previous model from {self.model_path}")
                del self.model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning while unloading model: {e}")
        self.model_path = selected_model_path
        self.load_model()
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.3) -> Iterator[tuple]:
        """流式生成播客音频。"""
        try:
            self.stop_generation = False
            self.is_generating = True
            if not script.strip():
                self.is_generating = False
                raise gr.Error("错误：请先输入脚本内容。")
            script = script.replace("’", "'")
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("错误：说话人数需在 1 到 4 之间。")
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"错误：请选择有效的第 {i+1} 位说话人。")
            log = f"🎙️ 正在生成包含 {num_speakers} 位说话人的播客\n"
            log += f"📊 参数：CFG={cfg_scale}，推理步数={self.inference_steps}\n"
            log += f"🎭 说话人：{', '.join(selected_speakers)}\n"
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 已停止生成", gr.update(visible=False)
                return
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"错误：无法加载 {speaker_name} 的音频样本。")
                voice_samples.append(audio_data)
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 已停止生成", gr.update(visible=False)
                return
            lines = script.strip().split('\n')
            formatted_script_lines = []
            # 预处理：支持 "Speaker 1:" / "speaker 2:" / "说话人1:" 等形式，并自动从 1 基改为 0 基
            import re
            temp_entries = []  # (orig_id or None, content)
            speaker_id_pattern = re.compile(r'^(Speaker|speaker|说话人)\s*(\d+)\s*[:：]\s*(.*)$')
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                m = speaker_id_pattern.match(line)
                if m:
                    sid = int(m.group(2))
                    content = m.group(3).strip()
                    temp_entries.append((sid, content))
                else:
                    temp_entries.append((None, line))  # 未指定说话人，后面轮询分配
            # 检测是否为 1 基编号（最小为1且不存在0）
            explicit_ids = [sid for sid, _ in temp_entries if sid is not None]
            shift = 0
            if explicit_ids and min(explicit_ids) == 1 and 0 not in explicit_ids:
                shift = -1  # 将 1..N 映射到 0..N-1
            # 生成标准化脚本
            auto_counter = 0
            for sid, content in temp_entries:
                if sid is None:
                    sid_out = auto_counter % num_speakers
                    auto_counter += 1
                else:
                    sid_out = sid + shift
                    if sid_out < 0:
                        sid_out = 0
                    if sid_out >= num_speakers:
                        sid_out = sid_out % num_speakers
                formatted_script_lines.append(f"Speaker {sid_out}: {content}")
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 脚本已格式化（共 {len(formatted_script_lines)} 轮）\n\n"
            log += "🔄 VibeVoice 处理中（流式模式）...\n"
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 已停止生成", gr.update(visible=False)
                return
            start_time = time.time()
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            self.current_streamer = audio_streamer
            generation_thread = threading.Thread(target=self._generate_with_streamer, args=(inputs, cfg_scale, audio_streamer))
            generation_thread.start()
            time.sleep(1)
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            sample_rate = 24000
            all_audio_chunks = []
            pending_chunks = []
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15
            min_chunk_size = sample_rate * 30
            audio_stream = audio_streamer.get_stream(0)
            has_yielded_audio = False
            has_received_chunks = False
            for audio_chunk in audio_stream:
                if self.stop_generation:
                    audio_streamer.end()
                    break
                chunk_count += 1
                has_received_chunks = True
                if torch.is_tensor(audio_chunk):
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                audio_16bit = convert_to_16_bit_wav(audio_np)
                all_audio_chunks.append(audio_16bit)
                pending_chunks.append(audio_16bit)
                pending_audio_size = sum(len(c) for c in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    should_yield = True; has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    should_yield = True
                if should_yield and pending_chunks:
                    new_audio = np.concatenate(pending_chunks)
                    total_duration = sum(len(c) for c in all_audio_chunks) / sample_rate
                    log_update = log + f"🎵 流式中：已生成 {total_duration:.1f}s（第 {chunk_count} 段）\n"
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    pending_chunks = []
                    last_yield_time = current_time
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(c) for c in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 最后一段流式：总时长 {total_duration:.1f}s\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True
            generation_thread.join(timeout=5.0)
            if generation_thread.is_alive():
                print("警告：生成线程超时")
                audio_streamer.end(); generation_thread.join(timeout=5.0)
            self.current_streamer = None; self.is_generating = False
            generation_time = time.time() - start_time
            if self.stop_generation:
                yield None, None, "🛑 已停止生成", gr.update(visible=False); return
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio)/sample_rate
                final_log = log + f"⏱️ 生成完成，用时 {generation_time:.2f} 秒\n🎵 最终音频时长：{final_duration:.2f} 秒\n📊 总分段：{chunk_count}\n✨ 生成成功！完整音频已就绪。\n💡 不满意？可以调整 CFG 重新生成。"
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False); return
            if not has_received_chunks:
                error_log = log + f"\n❌ 错误：未收到任何音频分段。总耗时 {generation_time:.2f}s"; yield None, None, error_log, gr.update(visible=False); return
            if not has_yielded_audio:
                error_log = log + f"\n❌ 错误：已生成音频但未成功流式输出。分段数：{chunk_count}"; yield None, None, error_log, gr.update(visible=False); return
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio)/sample_rate
                final_log = log + f"⏱️ 生成完成，用时 {generation_time:.2f} 秒\n🎵 最终音频时长：{final_duration:.2f} 秒\n📊 总分段：{chunk_count}\n✨ 生成成功！完整音频已在下方。\n💡 不满意？可以调整 CFG 重新生成。"
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
            else:
                final_log = log + "❌ 未生成任何音频。"; yield None, None, final_log, gr.update(visible=False)
        except gr.Error as e:
            self.is_generating = False; self.current_streamer = None
            err = f"❌ 输入错误：{str(e)}"; print(err); yield None, None, err, gr.update(visible=False)
        except Exception as e:
            self.is_generating = False; self.current_streamer = None
            err = f"❌ 未预期的错误：{str(e)}"; print(err); traceback.print_exc(); yield None, None, err, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
                
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        # Mirror path adjustment similar to voices directory: prefer demo/text_examples
        root_dir = os.path.dirname(__file__)
        candidate_paths = [
            os.path.join(root_dir, "demo", "text_examples"),
            os.path.join(root_dir, "text_examples"),
        ]
        examples_dir = None
        for p in candidate_paths:
            if os.path.exists(p):
                examples_dir = p
                break
        if examples_dir is None:
            examples_dir = candidate_paths[0]
        self.example_scripts = []
        
        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            
            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)
                
                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        # If no speakers found, default to 1
        if not speakers:
            return 1
        
        # Return the maximum speaker ID + 1 (assuming 0-based indexing)
        # or the count of unique speakers if they're 1-based
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            return max_speaker + 1
        else:
            # Assume 1-based indexing, return the count
            return len(speakers)
    

def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""
    
    custom_css = """
    :root { --accent:#10b981; --danger:#ef4444; --bg:#0f141b; --panel:#1c2530; --panel-border:#273341; --panel-border-strong:#324253; --text:#e5edf5; --text-dim:#92a5b8; --radius-lg:22px; --radius:14px; }
    .gradio-container { background: radial-gradient(circle at 30% 20%, #18202b 0%, #0f141b 60%, #0b0f14 100%); color: var(--text); font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    .main-header { background: linear-gradient(140deg,#18232f,#121a23); border:1px solid var(--panel-border); border-radius:28px; padding:2.1rem 2.4rem 2.5rem; margin:0 0 2.2rem; position:relative; overflow:hidden; box-shadow:0 8px 28px -8px rgba(0,0,0,.55),0 4px 10px -4px rgba(0,0,0,.5); }
    .main-header:before{content:"";position:absolute;inset:0;background:radial-gradient(circle at 25% 15%,rgba(16,185,129,.18),transparent 60%),radial-gradient(circle at 80% 70%,rgba(59,130,246,.2),transparent 55%);opacity:.9;}
    .main-header h1{margin:0 0 .85rem;font-size:2.35rem;font-weight:600;letter-spacing:.5px;background:linear-gradient(90deg,#6ee7b7,#3b82f6 55%,#8b5cf6);-webkit-background-clip:text;color:transparent;}
    .main-header p{margin:0;font-size:1.05rem;color:var(--text-dim);}
    .settings-card,.generation-card{background:linear-gradient(155deg,rgba(34,43,55,.92),rgba(25,32,41,.94));border:1px solid var(--panel-border);border-radius:var(--radius-lg);padding:1.45rem 1.4rem 1.6rem;box-shadow:0 4px 18px -6px rgba(0,0,0,.55);backdrop-filter:blur(10px) saturate(140%);}
    .settings-card:hover,.generation-card:hover{border-color:var(--panel-border-strong);box-shadow:0 8px 26px -10px rgba(0,0,0,.6),0 2px 6px -2px rgba(0,0,0,.55);}
    .speaker-item,.slider-container,.model-select{background:#1d252f;border:1px solid #2a3542;border-radius:var(--radius);padding:.85rem .9rem .9rem;}
    .speaker-item select,.model-select select{background:#25313d!important;color:var(--text)!important;border:1px solid #334252!important;border-radius:var(--radius)!important;}
    .model-select{margin-bottom:1rem;}
    .script-input,.log-output{background:#141c24!important;border:1px solid #2c3b49!important;border-radius:var(--radius)!important;color:var(--text)!important;font-family:'JetBrains Mono',ui-monospace,monospace!important;font-size:.88rem!important;}
    .script-input textarea::placeholder{color:#4b5b6b!important;}
    .generate-btn{background:linear-gradient(135deg,#10b981 0%,#0d9f73 55%,#0891b2 100%);border:none;border-radius:var(--radius);padding:.95rem 2rem;font-size:1.05rem;font-weight:600;color:#f0fdfa;letter-spacing:.5px;box-shadow:0 4px 18px -4px rgba(16,185,129,.55),0 2px 6px -2px rgba(0,0,0,.6);transition:all .25s ease;}
    .generate-btn:hover{transform:translateY(-2px);filter:brightness(1.08);box-shadow:0 8px 28px -6px rgba(16,185,129,.55),0 4px 14px -4px rgba(0,0,0,.55);}
    .stop-btn{background:linear-gradient(135deg,#ef4444 0%,#c53030 60%,#9b2c2c 100%);border:none;border-radius:var(--radius);padding:.95rem 1.6rem;font-size:1.02rem;font-weight:600;color:#ffecec;letter-spacing:.5px;box-shadow:0 4px 18px -4px rgba(239,68,68,.55),0 2px 6px -2px rgba(0,0,0,.55);transition:all .25s ease;}
    .random-btn{background:linear-gradient(135deg,#2f3944,#222b33);color:var(--text-dim);border:1px solid #364653;border-radius:var(--radius);padding:.9rem 1.4rem;font-weight:600;letter-spacing:.5px;transition:.25s ease;}
    .random-btn:hover{color:var(--text);border-color:#445768;background:linear-gradient(135deg,#374451,#252f38);}
    .streaming-status{background:linear-gradient(135deg,#1d2834,#161f27);border:1px solid #2c3a48;color:#6ee7b7;padding:.65rem 1.05rem;font-size:.78rem;font-weight:500;letter-spacing:.55px;border-radius:50px;display:inline-flex;align-items:center;gap:.55rem;box-shadow:0 3px 10px -3px rgba(0,0,0,.55);}
    .streaming-status .streaming-indicator{width:10px;height:10px;background:radial-gradient(circle at 30% 30%,#34d399,#059669);box-shadow:0 0 0 4px rgba(16,185,129,.15);border-radius:50%;animation:pulse 1.6s infinite;}
    @keyframes pulse{0%{transform:scale(1);opacity:1}50%{transform:scale(1.25);opacity:.55}100%{transform:scale(1);opacity:1}}
    .audio-output{background:linear-gradient(160deg,#141b23,#10161d);border:1px solid #27323f;border-radius:var(--radius-lg);padding:1.15rem 1.3rem;box-shadow:inset 0 0 0 1px rgba(255,255,255,.02),0 4px 16px -6px rgba(0,0,0,.6);}
    .complete-audio-section{margin-top:1rem;padding:1.05rem 1.15rem;background:linear-gradient(170deg,#10281f,#0d1c16);border:1px solid #1e4738;border-radius:var(--radius);box-shadow:inset 0 0 0 1px rgba(255,255,255,.03);}
    table{background:#161d25;border-collapse:separate;border-spacing:0;width:100%;overflow:hidden;border:1px solid #2a3644;border-radius:var(--radius);}
    table thead th{background:#1d252f;color:var(--text-dim);font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;padding:.6rem .7rem;}
    table tbody tr{transition:.18s ease;} table tbody tr:hover{background:#1f2935;} table td{border-top:1px solid #252f3b;padding:.65rem .8rem;font-size:.78rem;color:var(--text-dim);} table tbody tr:first-child td{border-top:none;}
    .markdown p{color:var(--text-dim);} .gradio-container .border{border:none!important;}
    """
    
    with gr.Blocks(
        title="VibeVoice - AI Podcast Generator",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Vibe 语音播客生成</h1>
            <p>使用 VibeVoice 生成多说话人长时播客音频</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ **播客配置**")
                # Model selection (lazy load on generate)
                # Discover available checkpoint dirs under checkpoints/
                project_root = os.path.dirname(__file__)
                ckpt_root = os.path.join(project_root, "checkpoints")
                model_dirs = []
                if os.path.exists(ckpt_root):
                    for d in sorted(os.listdir(ckpt_root)):
                        full = os.path.join(ckpt_root, d)
                        if os.path.isdir(full) and os.path.isfile(os.path.join(full, 'config.json')):
                            model_dirs.append(full)
                default_model_dir = demo_instance.model_path if demo_instance.model_path in model_dirs else (model_dirs[0] if model_dirs else "")
                model_selector = gr.Dropdown(
                    choices=model_dirs,
                    value=default_model_dir,
                    label="模型检查点路径",
                    interactive=True,
                    elem_classes="model-select"
                )
                
                # Number of speakers
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="说话人数",
                    elem_classes="slider-container"
                )
                
                # Speaker selection
                gr.Markdown("### 🎭 **说话人选择**")
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                # Choose first up to 4 voices as defaults (avoid hard-coded names that might not exist)
                default_speakers = available_speaker_names[:4]

                speaker_selections = []
                speaker_previews = []
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    speaker = gr.Dropdown(
                        choices=available_speaker_names,
                        value=default_value,
                        label=f"说话人 {i+1}",
                        visible=(i < 2),  # Initially show only first 2 speakers
                        elem_classes="speaker-item"
                    )
                    speaker_selections.append(speaker)
                    # 预览音频组件（文件路径模式，避免重复解码）
                    preview = gr.Audio(
                        label=f"预览 - 说话人 {i+1}",
                        type="filepath",
                        interactive=False,
                        visible=(i < 2),
                        show_download_button=False,
                        autoplay=False
                    )
                    speaker_previews.append(preview)
                
                # Advanced settings
                gr.Markdown("### ⚙️ **高级参数**")
                
                # Sampling parameters (contains all generation settings)
                with gr.Accordion("Generation Parameters", open=False):
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.3,
                        step=0.05,
                        label="CFG Scale（指导强度）",
                        # info="Higher values increase adherence to text",
                        elem_classes="slider-container"
                    )
                
            # Right column - Generation
            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 **脚本输入**")
                
                script_input = gr.Textbox(
                    label="对话脚本",
                    placeholder="""在此输入播客脚本，可使用以下格式：

Speaker 0: 大家好，欢迎来到我们的节目！
Speaker 1: 很高兴来到这里，今天我们来聊...

或直接粘贴纯文本，系统将自动轮流分配说话人。""",
                    lines=12,
                    max_lines=20,
                    elem_classes="script-input"
                )
                
                # Button row with Random Example on the left and Generate on the right
                with gr.Row():
                    # Random example button (now on the left)
                    random_example_btn = gr.Button(
                        "🎲 随机示例",
                        size="lg",
                        variant="secondary",
                        elem_classes="random-btn",
                        scale=1  # Smaller width
                    )
                    
                    # Generate button (now on the right)
                    generate_btn = gr.Button(
                        "🚀 开始生成",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2  # Wider than random button
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 停止生成",
                    size="lg",
                    variant="stop",
                    elem_classes="stop-btn",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Output section
                gr.Markdown("### 🎵 **生成结果**")
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="实时流式音频",
                    type="numpy",
                    elem_classes="audio-output",
                    streaming=True,  # Enable streaming mode
                    autoplay=True,
                    show_download_button=False,  # Explicitly show download button
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                complete_audio_output = gr.Audio(
                    label="完整音频（生成结束后可下载）",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    show_download_button=True,  # Explicitly show download button
                    visible=False  # Initially hidden, shown when audio is ready
                )
                
                gr.Markdown("""
                *💡 **流式音频**：边生成边播放，可能存在短暂停顿  
                *💡 **完整音频**：生成完成后在下方出现，可直接下载*
                """)
                
                # Generation log
                log_output = gr.Textbox(
                    label="生成日志",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )
        
        def update_speaker_visibility(num_speakers):
            dropdown_updates = []
            preview_updates = []
            for i in range(4):
                visible = i < num_speakers
                dropdown_updates.append(gr.update(visible=visible))
                preview_updates.append(gr.update(visible=visible))
            return dropdown_updates + preview_updates
        
        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_selections + speaker_previews
        )

        # 单个说话人下拉变化时更新对应预览
        def load_voice_preview(voice_name):
            if voice_name and voice_name in demo_instance.available_voices:
                return demo_instance.available_voices[voice_name]
            return None
        for i, dropdown in enumerate(speaker_selections):
            dropdown.change(
                fn=load_voice_preview,
                inputs=[dropdown],
                outputs=[speaker_previews[i]],
                queue=False
            )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(model_path, num_speakers, script, *speakers_and_params):
            """封装流式生成调用的包装函数。"""
            try:
                # Ensure correct model is loaded lazily
                if model_path:
                    yield None, gr.update(value=None, visible=False), f"🔄 正在加载模型：{model_path} ...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                    demo_instance.ensure_model_loaded(model_path)

                # Extract speakers and parameters
                speakers = speakers_and_params[:4]  # First 4 are speaker selections
                cfg_scale = speakers_and_params[4]   # CFG scale

                # Clear outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), "🎙️ 开始生成...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

                # The generator will yield multiple times
                final_log = "开始生成..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

            except Exception as e:
                error_msg = f"❌ 包装函数内部错误：{str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 已停止生成。", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting new generation."""
            return None, gr.update(value=None, visible=False)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[model_selector, num_speakers, script_input] + speaker_selections + [cfg_scale],
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
            queue=True  # Enable Gradio's built-in queue
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=lambda: (None, None),
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )
        
        # Function to randomly select an example
        def load_random_example():
            """Randomly select and load an example script."""
            import random
            
            # Get available examples
            if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
                example_scripts = demo_instance.example_scripts
            else:
                # Fallback to default
                example_scripts = [
                    [2, "Speaker 0: Welcome to our AI podcast demonstration!\nSpeaker 1: Thanks for having me. This is exciting!"]
                ]
            
            # Randomly select one
            if example_scripts:
                selected = random.choice(example_scripts)
                num_speakers_value = selected[0]
                script_value = selected[1]
                
                # Return the values to update the UI
                return num_speakers_value, script_value
            
            # Default values if no examples
            return 2, ""
        
        # Connect random example button
        random_example_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[num_speakers, script_input],
            queue=False  # Don't queue this simple operation
        )
        
        # Add usage tips
    # 使用说明（中文）
    gr.Markdown("""
    ### 💡 **使用提示**
    - 点击 **🚀 开始生成** 启动播客生成
    - 上方实时音频区域会边生成边播放
    - 生成完成后下方会出现完整音频，可下载保存
    - 生成过程中可点击 **🛑 停止生成** 立即中断
    - CFG Scale 越高越贴合文本，但可能降低多样性
    """)

    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/VibeVoice/checkpoints/VibeVoice-1.5B",
        help="Path to the VibeVoice model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for DDPM (not exposed to users)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ Initializing VibeVoice Demo with Streaming Support...")
    
    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port {args.port}")
    print(f"📁 Model path: {args.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        ).launch(
            share=False,
            server_port=args.port,
            server_name="0.0.0.0",
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()
