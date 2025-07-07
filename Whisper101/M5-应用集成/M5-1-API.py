"""
M5-1: Whisper API 服务

本脚本使用 FastAPI 构建一个语音转录微服务。
它提供一个 `/transcribe/` 端点，可以接收音频文件上传，
并返回转录后的全文文本和 SRT 格式的字幕。

如何运行:
1.  确保已安装所有依赖:
    pip install fastapi "uvicorn[standard]" python-multipart
2.  在终端中运行此脚本:
    uvicorn M5-1-API:app --reload --host 0.0.0.0 --port 8080
    
    - `--reload` 参数会在代码更改后自动重启服务，非常适合开发阶段。
3.  在浏览器中打开 http://127.0.0.1:8000/docs 可以看到自动生成的交互式 API 文档。
"""

import os
import whisper
import datetime
import torch
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 初始化 FastAPI 应用 ---
app = FastAPI(
    title="Whisper Transcription API",
    description="一个使用 OpenAI Whisper 进行语音转录和字幕生成的 API。",
    version="1.0.0"
)

# --- 全局变量与模型加载 ---
# 在生产环境中，推荐在应用启动时加载模型，避免每次请求都加载。
# 但对于开发和简单应用，按需加载也可以接受。
# 这里我们选择在函数内部加载，以支持动态选择模型。

# --- 辅助函数：SRT 生成器 (从 M4 稳定版复制) ---
def generate_srt_from_segments(segments):
    """根据 Whisper 的分段信息，生成 SRT 格式的字符串。"""
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = datetime.timedelta(seconds=int(segment['start']))
        end_time = datetime.timedelta(seconds=int(segment['end']))
        
        # 格式化时间为 SRT 标准格式: HH:MM:SS,ms
        start_str = str(start_time).split('.')[0] + f",{int((segment['start'] % 1) * 1000):03d}"
        end_str = str(end_time).split('.')[0] + f",{int((segment['end'] % 1) * 1000):03d}"
        
        text = segment['text'].strip()
        
        srt_content.append(f"{i + 1}")
        srt_content.append(f"{start_str} --> {end_str}")
        srt_content.append(f"{text}\n")
        
    return "\n".join(srt_content)

# --- API 端点定义 ---
@app.post("/transcribe/", 
          summary="转录音频文件", 
          description="上传一个音频文件，获取全文转录和SRT字幕。")
async def transcribe_audio(
    # 使用 File(...) 接收文件上传
    audio_file: UploadFile = File(..., description="要转录的音频文件 (如 .mp3, .wav, .m4a)"),
    # 使用 Form(...) 接收表单数据
    model_name: str = Form("base", description="要使用的 Whisper 模型 (如 'tiny', 'base', 'medium', 'large-v3')"),
    language: str = Form(None, description="音频的语言代码 (如 'en', 'zh')。如果留空，Whisper 会自动检测。")
):
    """
    处理音频转录请求的核心函数。
    """
    logging.info(f"接收到文件: {audio_file.filename} (类型: {audio_file.content_type})")
    logging.info(f"请求参数: model_name='{model_name}', language='{language}'")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"正在使用设备: {device}")
    
    # 使用临时文件来处理上传的音频
    # 这是处理上传文件的标准且安全的方式
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_audio_file:
            # 将上传的文件内容写入临时文件
            content = await audio_file.read()
            temp_audio_file.write(content)
            temp_audio_path = temp_audio_file.name
            logging.info(f"音频文件已临时保存到: {temp_audio_path}")

        # 加载 Whisper 模型
        logging.info(f"正在加载 Whisper 模型: {model_name}...")
        model = whisper.load_model(model_name, device=device)
        
        # 执行转录
        logging.info("开始转录...")
        start_time = datetime.datetime.now()
        
        # 如果 language 为 None 或空字符串，Whisper 会自动检测
        lang_option = language if language and language.strip() else None
        
        result = model.transcribe(temp_audio_path, language=lang_option, verbose=False)
        
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logging.info(f"转录完成，耗时: {processing_time:.2f} 秒")
        
        # 生成 SRT 字幕
        srt_subtitles = generate_srt_from_segments(result['segments'])
        
        # 准备返回的 JSON 数据
        response_data = {
            "full_text": result['text'],
            "srt_subtitles": srt_subtitles,
            "metadata": {
                "detected_language": result['language'],
                "processing_time_seconds": round(processing_time, 2),
                "model_used": model_name
            }
        }
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}", exc_info=True)
        # 如果发生任何错误，都返回一个 HTTP 500 错误
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        
    finally:
        # 确保临时文件在处理完成后被删除
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logging.info(f"已清理临时文件: {temp_audio_path}")

# --- 健康检查端点 ---
@app.get("/", summary="API 根目录与健康检查")
async def read_root():
    return {"message": "Whisper Transcription API 正在运行。请访问 /docs 查看 API 文档。"}


'''
测试和验证

curl -X POST "http://127.0.0.1:8080/transcribe/" \
     -F "audio_file=@M2-1-cn.wav;type=audio/wav" \
     -F "model_name=large-v3" \
     -F "language=zh"

'''