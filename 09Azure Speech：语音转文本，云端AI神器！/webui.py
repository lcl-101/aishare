import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from datetime import timedelta
import gradio as gr
import threading

# 加载环境变量
load_dotenv()

# 初始化语音配置
speech_key = os.getenv('SPEECH_KEY')
service_region = os.getenv('SPEECH_REGION')

if not speech_key or not service_region:
    raise ValueError("请设置 SPEECH_KEY 和 SPEECH_REGION 环境变量。")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "zh-CN"  # 设置语音识别语言为简体中文

def transcribe_audio(audio_file_path):
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    recognized_text = []
    srt_entries = []

    def recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text.append(evt.result.text)
            start_time = timedelta(seconds=evt.result.offset / 10**7)
            end_time = start_time + timedelta(seconds=evt.result.duration / 10**7)
            srt_entries.append((start_time, end_time, evt.result.text))

    def canceled(evt):
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"ErrorDetails={evt.error_details}")

    # 连接回调函数
    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.canceled.connect(canceled)

    # 开始连续识别
    speech_recognizer.start_continuous_recognition()

    # 等待识别完成
    done = threading.Event()
    def stop_cb(evt):
        done.set()

    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    done.wait()

    speech_recognizer.stop_continuous_recognition()

    # 返回识别的文本
    return ' '.join(recognized_text)

# 定义 Gradio 界面
def gradio_transcribe(audio):
    return transcribe_audio(audio)

iface = gr.Interface(
    fn=gradio_transcribe,
    inputs=gr.Audio(type="filepath", label="上传音频"),
    outputs="text",
    title="Azure Speech 音频转录",
    description="上传音频文件以将其转录为文本。"
)

# 启动界面
iface.launch()