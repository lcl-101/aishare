# Youtube 节目：
## Nvidia 开源AI“鹦鹉”Parakeet实测：英文语音秒转字幕，快到离谱！ 
## https://youtu.be/hMcaJy6hTGE

# 安装指南

## 安装 lfs 组件
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  
sudo apt install ffmpeg -y  
sudo apt install git-lfs -y  
git lfs install  

## 创建和激活运行环境
conda create -n Parakeet python=3.10 -y    
conda activate Parakeet  

## 克隆项目
git clone https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2  
mv parakeet-tdt-0.6b-v2 Parakeet  
cd Parakeet  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 获取样本数据
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav  

## 启动程序
python app.py   










 
















