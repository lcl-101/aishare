# Youtube 节目：
## “翻车”的 TTS？ZonosTTS 初体验：理想很丰满，现实很骨感！
## https://youtu.be/Lnf6SLFip20

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  
## 安装 cuda
wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run  
sudo bash cuda_12.4.1_550.54.15_linux.run  
rm cuda_12.4.1_550.54.15_linux.run  

sudo nano ~/.bashrc  
export PATH=/usr/local/cuda-12.4/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH  
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  
source ~/.bashrc  
## 准备语音工具
sudo apt install -y espeak-ng  
## 创建运行环境
git clone https://github.com/Zyphra/Zonos.git  
cd Zonos  
conda create -n Zonos python=3.10 -y    
conda activate Zonos  
pip install -e .  
pip install --no-build-isolation -e .[compile]  

## 推理
python gradio_interface.py  





