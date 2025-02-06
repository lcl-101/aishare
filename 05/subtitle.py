#使用 gradio 生成一个页面
#用户可以上传一个音频文件，上传之后，将由 whisper 处理，将文本返回给用户
#以下是 whisper 转写的示例代码



import os
import datetime
import torch
import subprocess
from silero_vad import load_silero_vad, read_audio,get_speech_timestamps
import re

from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

### 获取程序启动时间
demo_start_time = datetime.datetime.now()


#对于转写后的文本，我们可以定义一个关键词列表，用于替换文本中的关键词，可能是一些语气词
keywords = [
    ["呢", ""], 
    ["呢？", ""], 
    ["他", "它"], 
]

### 定义 model 路径
model_dir = "iic/SenseVoiceSmall"
### 加载 sensevoice 模型
model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

### 定义音视频文件目录
media_path = 'media'
### 定义音频、视频文件后缀
media_suffix = ['.wav', '.mp3', '.mp4', '.flv', '.avi', '.rmvb', '.mkv', '.wmv', '.rm', '.mov', '.3gp', '.mpeg', '.mpg', '.dat', '.asf', '.flac', '.aac', '.m4a', '.wma', '.ogg', '.ape', '.m4b', '.m4r', '.m4v', '.opus', '.webm', '.amr', '.ts', '.vob', '.wav', '.mp3', '.mp4', '.flv', '.avi', '.rmvb', '.mkv', '.wmv', '.rm', '.mov', '.3gp', '.mpeg', '.mpg', '.dat', '.asf', '.flac', '.aac', '.m4a', '.wma', '.ogg', '.ape', '.m4b', '.m4r', '.m4v', '.opus', '.webm', '.amr', '.ts', '.vob']

### 遍历音视频文件目录,并且对于每一个音视频文件生成文字保存在文本文件当中
for root, dirs, files in os.walk(media_path):
    for file in files:
        ### 定义一个空的 word_list 列表，用于存储转写的文字与时间戳信息
        vad_time_list = []
        ### 获取音视频文件的后缀
        media_file_suffix = os.path.splitext(file)[1]
        if media_file_suffix in media_suffix:
            media_file_path = os.path.join(root, file)
            srt_file_name = os.path.splitext(file)[0] + '.srt'
            srt_file_path = os.path.join(root, srt_file_name)
            # 打开字幕文件
            f = open(file=srt_file_path, mode="w", encoding='utf-8-sig')
            ### 将媒体文件转换为 wav 音频文件
            T_audio_file_name = "T-" + os.path.splitext(file)[0] + ".wav"
            T_audio_full_path = os.path.join(root, T_audio_file_name)
            ffmpeg_exe = 'ffmpeg.exe'
            ffmpeg_execute = subprocess.Popen([ffmpeg_exe, "-i", media_file_path, "-vn", "-y", T_audio_full_path])
            ffmpeg_code = ffmpeg_execute.wait()

            ### 使用 silero_vad 模型获取音频文件的时间戳信息
            vad_model = load_silero_vad()
            wav = read_audio(T_audio_full_path)
            vad_time = get_speech_timestamps(wav, vad_model, return_seconds=True)
            for i in range(len(vad_time)):
                if i==0:
                    vad_start_time = vad_time[i]["start"]
                    vad_end_time = vad_time[i + 1]["start"]

                elif i<len(vad_time)-1:
                    vad_start_time = vad_time[i]["start"]
                    vad_end_time = vad_time[i+1]["start"]
                else:
                    vad_start_time = vad_time[i]["start"]
                    vad_end_time = vad_time[i]["end"]
                
                srt_start_time = str(datetime.timedelta(seconds=vad_start_time))
                if i <= len(vad_time) - 2:
                    srt_end_time = (vad_time[i]["end"] + vad_time[i + 1]["start"]) / 2
                    srt_end_time = str(datetime.timedelta(seconds=vad_end_time))
                else:
                    srt_end_time = str(datetime.timedelta(seconds=vad_end_time))

                split_srt_start_time = srt_start_time.split('.')
                if len(split_srt_start_time) == 1:
                    srt_start_time = srt_start_time + ',000'
                else:
                    srt_start_time = (srt_start_time[:-3].replace('.', ','))

                split_srt_end_time = srt_end_time.split('.')
                if len(split_srt_end_time) == 1:
                    srt_end_time = srt_end_time + ',000'
                else:
                    srt_end_time = (srt_end_time[:-3].replace('.', ','))

                

                ### 基于 vad_start_time 和 vad_end_time，提取音频文件，然后使用 whisper 模型转写音频文件
                Seg_audio_file_name = "Seg-" + os.path.splitext(file)[0] + ".wav"
                Seg_audio_full_path = os.path.join(root, Seg_audio_file_name)

                ### 使用 ffmpeg 提取音频文件
                ffmpeg_exe = 'ffmpeg.exe'
                ffmpeg_execute = subprocess.Popen([ffmpeg_exe, "-i", T_audio_full_path, "-ss", str(vad_start_time), "-to", str(vad_end_time), "-y", Seg_audio_full_path])
                ffmpeg_code = ffmpeg_execute.wait()

                ### 使用 sensevoice 模型转写音频文件
                res = m.inference(
                    data_in=Seg_audio_full_path,
                    language="zh", # "zh", "en", "yue", "ja", "ko", "nospeech"
                    use_itn=False,
                    ban_emo_unk=False,
                    **kwargs,
                )

                text = rich_transcription_postprocess(res[0][0]["text"])
                print(text)
                #基于 keyword list 替换文本
                for keyword in keywords:
                    text = text.replace(keyword[0], keyword[1])
                
                # 使用正则表达式匹配中文和英文之间的位置，并在这些位置插入空格
                text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z])', r'\1 \2', text)
                text = re.sub(r'([a-zA-Z])([\u4e00-\u9fa5])', r'\1 \2', text)

                # 使用正则表达式将 ？符号替换为''
                text = text.replace("？", "")

                #写入 srt 文件
                f.write(str(i+1) + '\n')
                index_time = '{0} --> {1}'.format(srt_start_time, srt_end_time)
                f.write(index_time + '\n')
                f.write(text + '\n')
                f.write('\n')
                


                ### 删除 Seg 音频文件
                os.remove(Seg_audio_full_path)

        ### 删除 T 音频文件
        os.remove(T_audio_full_path)

        ### 关闭 srt 文件
        f.close()


### 获取程序结束时间
demo_end_time = datetime.datetime.now()
### 计算程序运行时间,以 00:00:00 格式输出
demo_run_time = demo_end_time - demo_start_time
### 打印程序运行时间
print("程序运行时间: ", demo_run_time)


