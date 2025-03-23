# Youtube 节目：
## 告别口型崩坏！LatentSync 1.5：字节跳动开源数字人，完美支持中文
## https://youtu.be/Elp_lEMcq2Q

# 安装指南

## 安装 ffmpeg
sudo apt install ffmpeg  
sudo apt -y install libgl1  

## 创建运行环境
conda create -y -n latentsync python=3.10.13  
conda activate latentsync  

## 克隆项目
git clone https://github.com/bytedance/LatentSync.git  
cd LatentSync  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型
huggingface-cli download ByteDance/LatentSync-1.5 --local-dir checkpoints --exclude "*.git*" "README.md"  

## 推理 
python gradio_app.py  










