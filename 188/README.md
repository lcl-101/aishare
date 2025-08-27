# Youtube 节目：
## 速度秒杀 Whisper！阿里最强开源语音转文字模型 SenseVoice 本地部署，一键生成精准字幕！
## https://youtu.be/oKTzRNzL9VQ

# 安装指南
## 创建运行环境
## 创建运行环境
conda create -n SenseVoice python=3.10 -y  
conda activate SenseVoice  

## 克隆文件
git clone https://github.com/FunAudioLLM/SenseVoice.git  
cd SenseVoice  

## 安装依赖组件
pip install -r requirements.txt  

## 准备模型文件
huggingface-cli download FunAudioLLM/SenseVoiceSmall --local-dir checkpoints/SenseVoiceSmall  

## 推理
python app.py  




  












 
















