# Youtube 节目：
## 只需一张图！MIDI 3D：AI快速生成3D模型，在线本地体验
## https://youtu.be/Gsjo4_lJIs4

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 创建运行环境
conda create -n midi3d python=3.10 -y  
conda activate midi3d  

## 克隆项目
git clone https://github.com/VAST-AI-Research/MIDI-3D.git  
cd MIDI-3D  

## 安装依赖组件
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  

## 下载模型
huggingface-cli download VAST-AI/MIDI-3D --local-dir checkpoints/MIDI-3D  
huggingface-cli download facebook/sam-vit-base --local-dir checkpoints/sam-vit-base  

## 推理 
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/anaconda3/envs/midi3d/lib/  
python gradio_demo.py  










