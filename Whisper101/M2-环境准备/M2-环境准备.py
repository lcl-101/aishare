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
wget http://cache.mixazure.com:88/whisper/Audio/M2-1-cn.wav -O ./audio/M2-1-cn.wav
whisper ./audio/M2-1-cn.wav --model large-v3 --language Chinese

wget http://cache.mixazure.com:88/whisper/Audio/M2-2-us.wav -O ./audio/M2-2-us.wav
whisper ./audio/M2-2-us.wav --model large-v3 --language English
wget http://cache.mixazure.com:88/whisper/Audio/139.wav -O ./audio/139.wav
rm M2-*.{json,srt,tsv,txt,vtt}

'''
'''
## 下载所有的模型
TARGET_DIR="models"
mkdir -p "$TARGET_DIR"

# tiny.en
echo "Downloading tiny.en..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt"

# tiny
echo "Downloading tiny..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"

# base.en
echo "Downloading base.en..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt"

# base
echo "Downloading base..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"

# small.en
echo "Downloading small.en..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt"

# small
echo "Downloading small..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"

# medium.en
echo "Downloading medium.en..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt"

# medium
echo "Downloading medium..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"

# large-v1
echo "Downloading large (v1)..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt"

# large-v2
echo "Downloading large (v2)..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"

# large-v3
echo "Downloading large (v3)..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"

# large-v3-turbo
echo "Downloading large (v3-turbo)..."
wget -c -P "$TARGET_DIR" "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt"

'''
'''
## 下载本地所有的模型
TARGET_DIR="models"
mkdir -p "$TARGET_DIR"

# tiny.en
echo "Downloading tiny.en..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/tiny.en.pt"

# tiny
echo "Downloading tiny..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/tiny.pt"

# base.en
echo "Downloading base.en..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/base.en.pt"

# base
echo "Downloading base..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/base.pt"

# small.en
echo "Downloading small.en..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/small.en.pt"

# small
echo "Downloading small..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/small.pt"

# medium.en
echo "Downloading medium.en..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/medium.en.pt"

# medium
echo "Downloading medium..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/medium.pt"

# large-v1
echo "Downloading large (v1)..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/large-v1.pt"

# large-v2
echo "Downloading large (v2)..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/large-v2.pt"

# large-v3
echo "Downloading large (v3)..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/large-v3.pt"

# large-v3-turbo
echo "Downloading large (v3-turbo)..."
wget -c -P "$TARGET_DIR" "http://cache.mixazure.com:88/whisper/models/large-v3-turbo.pt"

'''


## 代码测试
import os
import whisper
from whisper import load_model
from whisper.utils import get_writer
import datetime
import torch

def transcribe_audio(model_name, media_path, language="Chinese", save_txt=True, save_srt=True):
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
    
    # 保存结果到当前执行目录
    if save_txt:
        txt_writer = get_writer('txt', current_dir)
        txt_writer(result, media_path)
        print(f"TXT文件已保存到: {current_dir}")
    
    if save_srt:
        srt_writer = get_writer('srt', current_dir)
        srt_writer(result, media_path)
        print(f"SRT文件已保存到: {current_dir}")
    
    end_time = datetime.datetime.now()
    print(f'音视频文件：{media_path} --> 转写完成!')
    print(f"程序运行时间：{end_time - start_time}")
    
    return result

# 使用示例
if __name__ == "__main__":
    model_name = "large-v3"  # 使用模型名称而不是文件路径
    media_path = "../audio/M2-1-cn.wav"  # M2-环境准备目录到上级目录的audio文件夹
    
    result = transcribe_audio(model_name, media_path, language="Chinese")
    print("\n转录文本:")
    print(result["text"])