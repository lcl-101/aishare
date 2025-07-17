# Youtube 节目：
## AnimeGamer：腾讯开源AI，一句话生成动漫游戏场景！
## https://youtu.be/jb0eO2FJT6A

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

## 克隆项目
git clone https://github.com/gluttony-10/AnimeGamer.git  
cd AnimeGamer  

## 创建和激活运行环境
conda create -n AnimeGamer python=3.10 -y  
conda activate AnimeGamer  

## 安装依赖
pip install -r requirements.txt  

## 下载模型
huggingface-cli download TencentARC/AnimeGamer --local-dir checkpoints/AnimeGamer  
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --local-dir checkpoints/Mistral-7B-Instruct-v0.1  
huggingface-cli download Gluttony10/CogVideoX-2b-sat --local-dir checkpoints/CogVideoX-2b-sat   

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
python inference_MLLM.py  
python inference_Decoder.py   
















