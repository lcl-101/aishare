# 加载 speech SDK 和 dotenv
import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化语音配置
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
audio_config = speechsdk.audio.AudioOutputConfig(filename='demo.wav')

# 选择语音模型
speech_config.speech_synthesis_voice_name='zh-CN-YunjianNeural'

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# 读取 text 文件内容
text = open('demo.txt', 'r', encoding='utf-8').read()

# 语音合成，并且将结果导出到文件 demo.wav
speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()



# 输出结果


if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech synthesized for text [{}]".format(text))
elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = speech_synthesis_result.cancellation_details
    print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        if cancellation_details.error_details:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")