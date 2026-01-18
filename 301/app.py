"""
NeMo Streaming ASR Gradio Web Application
基于 NVIDIA Nemotron Speech Streaming 模型的语音识别 Web 应用
"""

import os
import tempfile
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
from omegaconf import OmegaConf

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
pipeline = None
asr_model = None

# 本地模型路径
MODEL_PATH = Path(__file__).parent / "checkpoints" / "nemotron-speech-streaming-en-0.6b" / "nemotron-speech-streaming-en-0.6b.nemo"

# 默认配置（内嵌在代码中，不需要单独配置文件）
DEFAULT_CONFIG = {
    # ASR 配置
    "asr": {
        "model_name": str(MODEL_PATH),  # 使用本地模型文件
        "device": "cuda",
        "device_id": 0,
        "compute_dtype": "bfloat16",
        "use_amp": True,
        "decoding": {
            "strategy": "greedy_batch",
            "preserve_alignments": False,
            "fused_batch_size": -1,
            "greedy": {
                "use_cuda_graph_decoder": False,
                "enable_per_stream_biasing": False,
                "max_symbols": 10,
                "ngram_lm_model": None,
                "ngram_lm_alpha": 0.0,
                "boosting_tree": {
                    "model_path": None,
                    "key_phrases_file": None,
                    "key_phrases_list": None,
                    "key_phrase_items_list": None,
                    "source_lang": "en",
                },
                "boosting_tree_alpha": 0.0,
            },
        },
    },
    # ITN 配置
    "itn": {
        "input_case": "lower_cased",
        "whitelist": None,
        "overwrite_cache": False,
        "max_number_of_permutations_per_split": 729,
        "left_padding_size": 4,
        "batch_size": 32,
        "n_jobs": 16,
    },
    # NMT 配置
    "nmt": {
        "model_name": "utter-project/EuroLLM-1.7B-Instruct",
        "source_language": "English",
        "target_language": "Russian",
        "waitk": -1,
        "device": "cuda",
        "device_id": 1,
        "batch_size": 16,
        "llm_params": {
            "dtype": "auto",
            "seed": 42,
        },
        "sampling_params": {
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 0.9,
            "seed": 42,
        },
    },
    # 置信度配置
    "confidence": {
        "exclude_blank": True,
        "aggregation": "mean",
        "method_cfg": {
            "name": "entropy",
            "entropy_type": "tsallis",
            "alpha": 0.5,
            "entropy_norm": "exp",
        },
    },
    # 端点检测配置
    "endpointing": {
        "stop_history_eou": 800,
        "residue_tokens_at_end": 2,
    },
    # 流式配置
    "streaming": {
        "sample_rate": 16000,
        "batch_size": 64,
        "word_boundary_tolerance": 4,
        "att_context_size": [70, 13],
        "use_cache": True,
        "use_feat_cache": True,
        "chunk_size_in_secs": None,
        "request_type": "frame",
        "num_slots": 256,
    },
    # Pipeline 设置
    "matmul_precision": "high",
    "log_level": 20,
    "pipeline_type": "cache_aware",
    "asr_decoding_type": "rnnt",
    # 运行时参数
    "audio_file": None,
    "output_filename": None,
    "output_dir": None,
    "enable_pnc": False,
    "enable_itn": False,
    "enable_nmt": False,
    "asr_output_granularity": "segment",
    "cache_dir": None,
    "lang": None,
    "return_tail_result": False,
    "calculate_wer": True,
    "calculate_bleu": True,
    # 指标配置
    "metrics": {
        "asr": {
            "gt_text_attr_name": "text",
            "clean_groundtruth_text": False,
            "langid": "en",
            "use_cer": False,
            "ignore_capitalization": True,
            "ignore_punctuation": True,
            "strip_punc_space": False,
        },
        "nmt": {
            "gt_text_attr_name": "answer",
            "ignore_capitalization": False,
            "ignore_punctuation": False,
            "strip_punc_space": False,
        },
    },
}


def get_config(
    enable_pnc: bool = False,
    enable_itn: bool = False,
    att_context_size: list = None
) -> OmegaConf:
    """获取配置对象"""
    cfg = OmegaConf.create(DEFAULT_CONFIG)
    
    # 更新运行时配置
    cfg.enable_pnc = enable_pnc
    cfg.enable_itn = enable_itn
    
    if att_context_size:
        cfg.streaming.att_context_size = att_context_size
    
    return cfg


def load_pipeline():
    """加载 ASR Pipeline"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    try:
        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
        
        logger.info("Loading config...")
        cfg = get_config()
        
        logger.info("Building ASR pipeline...")
        pipeline = PipelineBuilder.build_pipeline(cfg)
        logger.info("Pipeline loaded successfully!")
        
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise


def load_simple_model():
    """加载简单的 ASR 模型（不使用 pipeline）"""
    global asr_model
    
    if asr_model is not None:
        return asr_model
    
    try:
        import nemo.collections.asr as nemo_asr
        
        logger.info(f"Loading ASR model from {MODEL_PATH}...")
        # 使用本地模型文件
        asr_model = nemo_asr.models.ASRModel.restore_from(str(MODEL_PATH))
        logger.info("Model loaded successfully!")
        
        return asr_model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def transcribe_audio_file(
    audio_path: str,
    use_pipeline: bool = True,
    enable_pnc: bool = False,
    enable_itn: bool = False,
    att_context_size: str = "[70, 13]"
) -> Tuple[str, str]:
    """
    转录音频文件
    
    Args:
        audio_path: 音频文件路径
        use_pipeline: 是否使用 pipeline 模式
        enable_pnc: 是否启用标点和大小写
        enable_itn: 是否启用逆文本规范化
        att_context_size: 注意力上下文大小配置
        
    Returns:
        (转录文本, 推理时间信息)
    """
    if audio_path is None:
        return "请上传音频文件", ""
    
    try:
        start_time = time.time()
        
        if use_pipeline:
            # 使用 Pipeline 模式
            from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
            
            # 解析 att_context_size
            context_size = None
            try:
                parsed = eval(att_context_size)
                if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                    context_size = list(parsed)
            except:
                pass
            
            # 获取配置
            cfg = get_config(
                enable_pnc=enable_pnc,
                enable_itn=enable_itn,
                att_context_size=context_size
            )
            
            # 构建 pipeline 并运行
            pipe = PipelineBuilder.build_pipeline(cfg)
            output = pipe.run([audio_path])
            
            inference_time = time.time() - start_time
            
            # 提取文本 - Pipeline 返回格式: {file_index: {'text': ..., 'segments': [...], ...}}
            if output is None:
                return "未检测到语音内容", f"⏱️ 推理时间: {inference_time:.2f} 秒"
            
            results = []
            
            if isinstance(output, dict):
                # Pipeline 返回 {0: {'text': '...', 'segments': [...], ...}, 1: {...}, ...}
                for file_idx, file_result in output.items():
                    if isinstance(file_result, dict) and 'text' in file_result:
                        text = file_result['text']
                        if text and isinstance(text, str) and text.strip():
                            results.append(text.strip())
                    elif isinstance(file_result, str) and file_result.strip():
                        results.append(file_result.strip())
            elif isinstance(output, str) and output.strip():
                results.append(output.strip())
            elif hasattr(output, 'text'):
                results.append(str(output.text).strip())
            
            transcription = "\n".join(results) if results else "未检测到语音内容"
            
            # 计算音频时长（如果可能）
            try:
                import librosa
                duration = librosa.get_duration(path=audio_path)
                rtf = inference_time / duration  # Real-Time Factor
                time_info = f"⏱️ 推理时间: {inference_time:.2f} 秒 | 音频时长: {duration:.2f} 秒 | RTF: {rtf:.3f}x"
            except:
                time_info = f"⏱️ 推理时间: {inference_time:.2f} 秒"
            
            return transcription, time_info
        
        else:
            # 使用简单模型模式
            model = load_simple_model()
            transcription = model.transcribe([audio_path])
            
            inference_time = time.time() - start_time
            
            if isinstance(transcription, list) and len(transcription) > 0:
                if isinstance(transcription[0], str):
                    result = transcription[0]
                elif hasattr(transcription[0], 'text'):
                    result = transcription[0].text
                else:
                    result = str(transcription)
            else:
                result = str(transcription)
            
            try:
                import librosa
                duration = librosa.get_duration(path=audio_path)
                rtf = inference_time / duration
                time_info = f"⏱️ 推理时间: {inference_time:.2f} 秒 | 音频时长: {duration:.2f} 秒 | RTF: {rtf:.3f}x"
            except:
                time_info = f"⏱️ 推理时间: {inference_time:.2f} 秒"
            
            return result, time_info
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"转录错误: {str(e)}", ""


def transcribe_microphone(
    audio: Optional[Tuple[int, np.ndarray]],
    use_pipeline: bool = True,
    enable_pnc: bool = False,
    enable_itn: bool = False,
    att_context_size: str = "[70, 13]"
) -> Tuple[str, str]:
    """
    转录麦克风录音
    
    Args:
        audio: 麦克风录音数据 (sample_rate, audio_data)
        use_pipeline: 是否使用 pipeline 模式
        enable_pnc: 是否启用标点和大小写
        enable_itn: 是否启用逆文本规范化
        att_context_size: 注意力上下文大小配置
        
    Returns:
        (转录文本, 推理时间信息)
    """
    if audio is None:
        return "请录制音频", ""
    
    try:
        import scipy.io.wavfile as wav
        
        sample_rate, audio_data = audio
        
        # 确保音频是单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # 确保是 16kHz
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            sample_rate = 16000
        
        # 归一化音频
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            wav.write(temp_path, 16000, (audio_data * 32767).astype(np.int16))
        
        # 转录
        result, time_info = transcribe_audio_file(
            temp_path, 
            use_pipeline, 
            enable_pnc, 
            enable_itn,
            att_context_size
        )
        
        # 清理临时文件
        os.unlink(temp_path)
        
        return result, time_info
        
    except Exception as e:
        logger.error(f"Microphone transcription error: {e}")
        return f"转录错误: {str(e)}", ""


def get_latency_info(att_context_size: str) -> str:
    """获取延迟信息"""
    latency_map = {
        "[70, 0]": "Chunk size = 1 (1 × 80ms = 0.08s) - 最低延迟",
        "[70, 1]": "Chunk size = 2 (2 × 80ms = 0.16s)",
        "[70, 6]": "Chunk size = 7 (7 × 80ms = 0.56s)",
        "[70, 13]": "Chunk size = 14 (14 × 80ms = 1.12s) - 最高精度",
    }
    return latency_map.get(att_context_size, "自定义配置")


# 自定义 CSS
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
}
.title {
    text-align: center;
    margin-bottom: 20px;
}
"""

# 创建 Gradio 界面
def create_app():
    """创建 Gradio 应用"""
    
    with gr.Blocks(title="NeMo Streaming ASR") as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 10px;">
            <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="text-decoration: none; color: #ff0000;">
                📺 <strong>AI 技术分享频道</strong> - 欢迎订阅我的 YouTube 频道！
            </a>
        </div>
        <div class="title">
            <h1>🎙️ NVIDIA NeMo Streaming ASR</h1>
            <p>基于 Nemotron Speech Streaming 0.6B 模型的实时语音识别</p>
        </div>
        """)
        
        with gr.Tabs():
            # 文件上传标签页
            with gr.TabItem("📁 文件上传", id="file_upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.Audio(
                            label="上传音频文件",
                            type="filepath",
                            sources=["upload"],
                        )
                        
                        with gr.Accordion("⚙️ 高级设置", open=False):
                            file_use_pipeline = gr.Checkbox(
                                label="使用 Pipeline 模式",
                                value=True,
                                info="Pipeline 模式支持更多功能（PnC、ITN 等）"
                            )
                            file_enable_pnc = gr.Checkbox(
                                label="启用标点和大小写 (PnC)",
                                value=False,
                                info="自动添加标点符号和正确的大小写"
                            )
                            file_enable_itn = gr.Checkbox(
                                label="启用逆文本规范化 (ITN)",
                                value=False,
                                info="将口语数字转换为阿拉伯数字等"
                            )
                            file_context_size = gr.Dropdown(
                                label="延迟配置 (att_context_size)",
                                choices=["[70, 0]", "[70, 1]", "[70, 6]", "[70, 13]"],
                                value="[70, 13]",
                                info="较大的右上下文提供更高精度但增加延迟"
                            )
                            file_latency_info = gr.Textbox(
                                label="延迟信息",
                                value="Chunk size = 14 (14 × 80ms = 1.12s) - 最高精度",
                                interactive=False
                            )
                        
                        file_submit_btn = gr.Button("🚀 开始转录", variant="primary")
                    
                    with gr.Column(scale=1):
                        file_output = gr.Textbox(
                            label="转录结果",
                            lines=10,
                            placeholder="转录结果将显示在这里..."
                        )
                        file_time_info = gr.Textbox(
                            label="推理统计",
                            interactive=False,
                            placeholder="推理时间将显示在这里..."
                        )
                
                # 事件绑定
                file_context_size.change(
                    fn=get_latency_info,
                    inputs=[file_context_size],
                    outputs=[file_latency_info]
                )
                
                file_submit_btn.click(
                    fn=transcribe_audio_file,
                    inputs=[
                        file_input, 
                        file_use_pipeline, 
                        file_enable_pnc, 
                        file_enable_itn,
                        file_context_size
                    ],
                    outputs=[file_output, file_time_info]
                )
            
            # 麦克风录音标签页
            with gr.TabItem("🎤 麦克风录音", id="microphone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mic_input = gr.Audio(
                            label="录制音频",
                            type="numpy",
                            sources=["microphone"],
                        )
                        
                        with gr.Accordion("⚙️ 高级设置", open=False):
                            mic_use_pipeline = gr.Checkbox(
                                label="使用 Pipeline 模式",
                                value=True,
                                info="Pipeline 模式支持更多功能（PnC、ITN 等）"
                            )
                            mic_enable_pnc = gr.Checkbox(
                                label="启用标点和大小写 (PnC)",
                                value=False,
                                info="自动添加标点符号和正确的大小写"
                            )
                            mic_enable_itn = gr.Checkbox(
                                label="启用逆文本规范化 (ITN)",
                                value=False,
                                info="将口语数字转换为阿拉伯数字等"
                            )
                            mic_context_size = gr.Dropdown(
                                label="延迟配置 (att_context_size)",
                                choices=["[70, 0]", "[70, 1]", "[70, 6]", "[70, 13]"],
                                value="[70, 13]",
                                info="较大的右上下文提供更高精度但增加延迟"
                            )
                            mic_latency_info = gr.Textbox(
                                label="延迟信息",
                                value="Chunk size = 14 (14 × 80ms = 1.12s) - 最高精度",
                                interactive=False
                            )
                        
                        mic_submit_btn = gr.Button("🚀 开始转录", variant="primary")
                    
                    with gr.Column(scale=1):
                        mic_output = gr.Textbox(
                            label="转录结果",
                            lines=10,
                            placeholder="转录结果将显示在这里..."
                        )
                        mic_time_info = gr.Textbox(
                            label="推理统计",
                            interactive=False,
                            placeholder="推理时间将显示在这里..."
                        )
                
                # 事件绑定
                mic_context_size.change(
                    fn=get_latency_info,
                    inputs=[mic_context_size],
                    outputs=[mic_latency_info]
                )
                
                mic_submit_btn.click(
                    fn=transcribe_microphone,
                    inputs=[
                        mic_input, 
                        mic_use_pipeline, 
                        mic_enable_pnc, 
                        mic_enable_itn,
                        mic_context_size
                    ],
                    outputs=[mic_output, mic_time_info]
                )
        
        # 使用说明
        gr.Markdown("""
### 📖 使用说明

- **支持的音频格式:** WAV, MP3, FLAC, OGG 等常见格式
- **采样率要求:** 16kHz（其他采样率会自动转换）
- **最小音频长度:** 至少 80ms
- **输出:** 英文文本转录，支持标点和大小写

#### 🔧 延迟配置说明

| 配置 | Chunk 大小 | 延迟 | 说明 |
|:---:|:---:|:---:|:---:|
| [70, 0] | 1 帧 | 0.08s | 最低延迟 |
| [70, 1] | 2 帧 | 0.16s | 低延迟 |
| [70, 6] | 7 帧 | 0.56s | 平衡模式 |
| [70, 13] | 14 帧 | 1.12s | 最高精度 |
        """)
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; color: #666;">
            <p>Powered by <a href="https://developer.nvidia.com/nemo" target="_blank">NVIDIA NeMo</a> | 
            Model: <a href="https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b" target="_blank">Nemotron Speech Streaming 0.6B</a></p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    # 预加载模型
    logger.info("预加载 ASR Pipeline...")
    try:
        load_pipeline()
        logger.info("模型预加载完成！")
    except Exception as e:
        logger.warning(f"预加载失败，将在首次使用时加载: {e}")
    
    # 创建并启动应用
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )
