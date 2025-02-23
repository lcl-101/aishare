# Youtube 节目：
## Google 开源 PaliGemma2 体验：视觉语言模型，图像问答、分割全能！
## https://youtu.be/6gGMxtCMyWA

# 安装指南

## 安装 lfs
sudo apt-get update
sudo apt-get install git
sudo apt-get install git-lfs

## 创建 python 运行环境
cd /home/softice
conda create -n paligemma2 python=3.10.0 -y
conda activate paligemma2

## 克隆项目和下载模型
git clone https://huggingface.co/spaces/google/paligemma2-10b-mix
cd paligemma2-10b-mix

## 安装依赖组件
pip install -r requirements.txt
pip install gradio

## 推理
python app.py






