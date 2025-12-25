# Youtube 节目：
## 免费平替 GPT-4o 语音模式？阿里开源 Fun-Audio-Chat：本地搭建懂情感的全能 AI 助手！
## https://youtu.be/5-hrOcG0lqA

# 安装指南
## 克隆项目
git clone --recurse-submodules https://github.com/FunAudioLLM/Fun-Audio-Chat.git  
cd Fun-Audio-Chat  

## 创建运行环境
conda create -n fun-audio-chat python=3.10 -y  
conda activate fun-audio-chat  

## 安装依赖组件
sudo apt install ffmpeg  
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
pip install huggingface-hub  
pip install -r requirements.txt  
pip install gradio  

## 模型下载
hf download FunAudioLLM/Fun-Audio-Chat-8B --local-dir checkpoints/Fun-Audio-Chat-8B  
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir checkpoints/Fun-CosyVoice3-0.5B-2512  

## 推理演示
python app.py        

  












 
















