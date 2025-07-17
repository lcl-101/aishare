# Youtube 节目：
## EdgeTAM：Facebook开源AI，手机也能实时识别视频物体！
## https://youtu.be/E4ClWAQOdxA

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

## 创建和激活运行环境
conda create -n EdgeTAM python=3.10 -y    
conda activate EdgeTAM  

## 安装 Pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  

## 克隆项目
git clone https://github.com/facebookresearch/EdgeTAM.git && cd EdgeTAM  

## 安装依赖组件
pip install -e .  
pip install -r requirements.txt  

## 启动程序
python gradio_app.py  









 
















