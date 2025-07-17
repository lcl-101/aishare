# Youtube 节目：
## AI 解读“第一人称”世界！EgoGPT 体验：用大模型看懂你的智能眼镜视频！
## https://youtu.be/x_nBPE4h44Q

# 安装指南

## 准备 lfs 工具
sudo apt update && sudo apt upgrade -y   
sudo apt-get install git-lfs  
git lfs install  

## 准备 ffmpeg 组件
sudo apt install ffmpeg -y  

## 准备编译工具
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
conda create -n EgoGPT-7B python=3.10.0 -y  
conda activate EgoGPT-7B  

## 克隆项目
git clone https://huggingface.co/spaces/Jingkang/EgoGPT-7B  
cd EgoGPT-7B  

## 安装依赖组件
pip install -r requirements.txt  
pip install flash-attn --no-build-isolation  
pip install pyav  

## 下载模型文件
mkdir checkpoints  
huggingface-cli download lmms-lab/EgoGPT-7b-Demo --local-dir checkpoints/EgoGPT-7b-Demo   
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir checkpoints/siglip-so400m-patch14-384  
## 推理 
python app.py  








