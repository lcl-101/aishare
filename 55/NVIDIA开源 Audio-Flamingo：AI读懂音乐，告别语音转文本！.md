# Youtube 节目：
## YOLOE NVIDIA开源 Audio-Flamingo：AI读懂音乐，告别语音转文本！
## https://youtu.be/zyyCGJoZEzg

# 安装指南

## 安装系统组件
sudo apt install git-lfs  
git lfs install  

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

## 创建运行环境
conda create -n flamingo python=3.10 -y  
conda activate flamingo  

## 克隆项目
git clone https://huggingface.co/spaces/nvidia/audio-flamingo-2  
cd audio-flamingo-2  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 推理 
python app.py  









