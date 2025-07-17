# Youtube 节目：
## Kimi-Audio：AI语音转写、理解、对话，功能强大！
## https://youtu.be/XdV9TNngy0M

# 安装指南

## 准备系统工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  
sudo apt-get install ninja-build -y  
sudo apt install ffmpeg -y  

## 安装 cuda
wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run  
sudo bash cuda_12.4.1_550.54.15_linux.run  

rm cuda_12.4.1_550.54.15_linux.run  
sudo nano ~/.bashrc  

export PATH=/usr/local/cuda-12.4/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH  
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  

source ~/.bashrc  
## 克隆项目
git clone --recurse-submodules https://github.com/MoonshotAI/Kimi-Audio.git  
cd Kimi-Audio  

## 创建和激活运行环境
conda create -n KimiAudio python=3.10 -y    
conda activate KimiAudio  

## 安装依赖组件
pip install -r requirements.txt  
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" --no-deps  
pip install gradio  

## 下载模型
pip install "huggingface_hub[cli]"  
huggingface-cli download moonshotai/Kimi-Audio-7B-Instruct --local-dir checkpoints/Kimi-Audio-7B-Instruct  
huggingface-cli download THUDM/glm-4-voice-tokenizer --local-dir checkpoints/glm-4-voice-tokenizer  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6  
python app.py  










 
















