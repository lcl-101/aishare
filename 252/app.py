"""
VibeVoice Realtime TTS - FastAPI + WebSocket 后端
基于官方 demo 简化整合
"""

import argparse
import asyncio
import copy
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

# ============ 配置 ============
SAMPLE_RATE = 24_000
PROJECT_ROOT = Path(__file__).parent
LOCAL_TOKENIZER_PATH = PROJECT_ROOT / "checkpoints" / "Qwen2.5-0.5B"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "VibeVoice-Realtime-0.5B"
VOICES_DIR = PROJECT_ROOT / "demo" / "voices" / "streaming_model"


class TTSService:
    """TTS 服务"""

    def __init__(self, model_path: Path, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._torch_device = torch.device(device)
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self._voice_cache: Dict[str, Any] = {}

    def load(self) -> None:
        """加载模型"""
        print(f"[启动] 加载处理器: {self.model_path}")
        
        tokenizer_path = str(LOCAL_TOKENIZER_PATH) if LOCAL_TOKENIZER_PATH.exists() else "Qwen/Qwen2.5-0.5B"
        print(f"[启动] Tokenizer: {tokenizer_path}")
        
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(
            str(self.model_path),
            language_model_pretrained_name=tokenizer_path,
        )

        # 设备配置
        if self.device == "cuda":
            load_dtype, device_map, attn_impl = torch.bfloat16, "cuda", "flash_attention_2"
        elif self.device == "mps":
            load_dtype, device_map, attn_impl = torch.float32, None, "sdpa"
        else:
            load_dtype, device_map, attn_impl = torch.float32, "cpu", "sdpa"

        print(f"[启动] 设备: {self.device}, dtype: {load_dtype}, attn: {attn_impl}")

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                str(self.model_path),
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
        except Exception:
            if attn_impl == "flash_attention_2":
                print("[警告] Flash Attention 2 失败，使用 SDPA")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    str(self.model_path),
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
            else:
                raise

        if self.device == "mps":
            self.model.to("mps")

        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=5)

        # 加载声音预设
        self._load_voices()
        print("[启动] 模型加载完成!")

    def _load_voices(self) -> None:
        """加载声音预设"""
        if not VOICES_DIR.exists():
            print(f"[警告] 声音目录不存在: {VOICES_DIR}")
            return
        for pt in VOICES_DIR.glob("*.pt"):
            self.voice_presets[pt.stem] = pt
        print(f"[启动] 声音预设: {len(self.voice_presets)} 个")

    def get_voice(self, key: str) -> Any:
        """获取声音预设（带缓存）"""
        if key not in self._voice_cache:
            path = self.voice_presets.get(key)
            if not path:
                key = next(iter(self.voice_presets))
                path = self.voice_presets[key]
            self._voice_cache[key] = torch.load(path, map_location=self._torch_device, weights_only=False)
        return self._voice_cache[key]

    def stream(
        self,
        text: str,
        voice_key: str,
        cfg_scale: float = 1.5,
        inference_steps: int = 5,
        stop_event: Optional[threading.Event] = None,
    ):
        """流式生成语音，yield PCM16 bytes"""
        if not text.strip():
            return

        text = text.replace("'", "'")
        prefilled = self.get_voice(voice_key)
        self.model.set_ddpm_inference_steps(num_steps=inference_steps)

        # 准备输入
        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip(),
            cached_prompt=prefilled,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(self._torch_device) if hasattr(v, "to") else v for k, v in processed.items()}

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        stop = stop_event or threading.Event()
        errors = []

        def run():
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False, "temperature": 1.0, "top_p": 1.0},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop.is_set,
                    verbose=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(prefilled),
                )
            except Exception as e:
                errors.append(e)
            finally:
                audio_streamer.end()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        try:
            for chunk in audio_streamer.get_stream(0):
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)

                # 归一化并转 PCM16
                peak = np.max(np.abs(chunk))
                if peak > 1.0:
                    chunk = chunk / peak
                pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                yield pcm16.tobytes()
        finally:
            stop.set()
            thread.join(timeout=5)


# ============ FastAPI 应用 ============
app = FastAPI(title="VibeVoice Realtime TTS")
tts_service: Optional[TTSService] = None
ws_lock = asyncio.Lock()


@app.on_event("startup")
async def startup():
    global tts_service
    model_path = getattr(app.state, "model_path", DEFAULT_MODEL_PATH)
    device = getattr(app.state, "device", "cuda")
    
    tts_service = TTSService(model_path=Path(model_path), device=device)
    tts_service.load()


@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/config")
async def config():
    """返回配置信息"""
    voices = sorted(tts_service.voice_presets.keys()) if tts_service else []
    return JSONResponse({
        "voices": voices,
        "default_voice": voices[0] if voices else None,
        "sample_rate": SAMPLE_RATE,
    })


@app.websocket("/stream")
async def websocket_stream(ws: WebSocket):
    """WebSocket 流式 TTS"""
    await ws.accept()
    
    # 解析参数
    text = ws.query_params.get("text", "")
    voice = ws.query_params.get("voice", "")
    cfg = float(ws.query_params.get("cfg", "1.5"))
    steps = int(ws.query_params.get("steps", "5"))

    # 检查是否繁忙
    if ws_lock.locked():
        await ws.send_text(json.dumps({"type": "error", "message": "服务繁忙，请稍后再试"}))
        await ws.close(code=1013)
        return

    async with ws_lock:
        stop_event = threading.Event()
        start_time = time.time()
        total_samples = 0
        first_chunk = True

        try:
            # 发送开始信号
            await ws.send_text(json.dumps({"type": "start", "text": text}))

            # 流式生成
            iterator = tts_service.stream(
                text=text,
                voice_key=voice,
                cfg_scale=cfg,
                inference_steps=steps,
                stop_event=stop_event,
            )

            for pcm_bytes in iterator:
                if ws.client_state != WebSocketState.CONNECTED:
                    break

                # 发送音频数据
                await ws.send_bytes(pcm_bytes)
                
                samples = len(pcm_bytes) // 2
                total_samples += samples

                if first_chunk:
                    first_chunk = False
                    latency = time.time() - start_time
                    await ws.send_text(json.dumps({
                        "type": "info",
                        "first_chunk_latency": round(latency, 3),
                    }))

                # 发送进度
                duration = total_samples / SAMPLE_RATE
                await ws.send_text(json.dumps({
                    "type": "progress",
                    "generated_seconds": round(duration, 2),
                }))

            # 发送完成信号
            total_time = time.time() - start_time
            await ws.send_text(json.dumps({
                "type": "done",
                "total_seconds": round(total_samples / SAMPLE_RATE, 2),
                "total_time": round(total_time, 2),
            }))

        except WebSocketDisconnect:
            print("[WS] 客户端断开")
            stop_event.set()
        except Exception as e:
            print(f"[WS] 错误: {e}")
            stop_event.set()
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except:
                pass
        finally:
            stop_event.set()
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()


def main():
    parser = argparse.ArgumentParser(description="VibeVoice Realtime TTS Server")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app.state.model_path = args.model_path
    app.state.device = args.device

    print("=" * 50)
    print("VibeVoice Realtime TTS Server")
    print(f"端口: {args.port}")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
