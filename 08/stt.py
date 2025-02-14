import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from datetime import timedelta

# 加载环境变量
load_dotenv()

# 定义音频文件、文本文件和字幕文件的路径
audio_file_path = 'short.wav'
txt_file_path = 'output.txt'
srt_file_path = 'output.srt'

# 如果 txt 和 srt 文件存在，删除它们
if os.path.exists(txt_file_path):
    os.remove(txt_file_path)
if os.path.exists(srt_file_path):
    os.remove(srt_file_path)

# 初始化语音配置
speech_key = os.getenv('SPEECH_KEY')
service_region = os.getenv('SPEECH_REGION')

if not speech_key or not service_region:
    raise ValueError("请设置 SPEECH_KEY 和 SPEECH_REGION 环境变量。")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "zh-CN"  # 设置语音识别语言为简体中文

audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# 定义回调函数来处理识别结果
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
import time
done = False
def stop_cb(evt):
    global done
    done = True

speech_recognizer.session_stopped.connect(stop_cb)
speech_recognizer.canceled.connect(stop_cb)

while not done:
    time.sleep(0.5)

speech_recognizer.stop_continuous_recognition()

# 保存识别的文本到 txt 文件
with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    txt_file.write(' '.join(recognized_text))

# 保存识别的文本到 srt 文件
with open(srt_file_path, 'w', encoding='utf-8') as srt_file:
    for i, (start, end, text) in enumerate(srt_entries):
        srt_file.write(f"{i+1}\n")
        srt_file.write(f"{str(start)[:-3].replace('.', ',')} --> {str(end)[:-3].replace('.', ',')}\n")
        srt_file.write(text + "\n\n")