'''
mkdir whisper
mkdir whisper/audio
sudo apt install ffmpeg

## 创建和激活运行环境
conda create -n whisper python=3.10 -y
conda activate whisper

## 安装 openai whisper
pip install -U openai-whisper

## 测试效果
wget http://cache.mixazure.com:88/whisper/Audio/00000034.wav -O ./audio/00000034.wav

'''


## 代码测试
import os
import whisper
from whisper import load_model
from whisper.utils import get_writer
import datetime
import torch

def transcribe_audio(model_name, media_path, language="Chinese"):
    """
    使用Whisper模型转录音频文件
    
    Args:
        model_name (str): 模型名称 (如: 'large-v3', 'base', 'small' 等)
        media_path (str): 音频文件路径
        language (str): 转录语言，默认为"Chinese"
        save_txt (bool): 是否保存txt文件，默认为True
        save_srt (bool): 是否保存srt字幕文件，默认为True
    
    Returns:
        dict: 转录结果
    """
    start_time = datetime.datetime.now()
    
    # 获取当前脚本执行目录作为保存路径
    current_dir = os.getcwd()  # 获取当前工作目录
    
    # 定义设备,如果有GPU则使用GPU,否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_name}")
    model = whisper.load_model(model_name, device=device, in_memory=True)
    
    # 转录音频
    print(f"开始转录: {media_path}")
    result = model.transcribe(media_path, fp16=False, language=language, verbose=True)
    
    end_time = datetime.datetime.now()
    print(f'音视频文件：{media_path} --> 转写完成!')
    print(f"程序运行时间：{end_time - start_time}")
    
    return result

# 使用示例
if __name__ == "__main__":
    model_name = "large-v3"  # 使用模型名称而不是文件路径
    media_path = "../audio/00000034.wav"  # 修正路径，从M2-环境准备目录到上级目录的audio文件夹
    
    result = transcribe_audio(model_name, media_path, language="Chinese")
    print("\n转录文本:")
    print(result["text"])