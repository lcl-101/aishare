# Youtube 节目：
##  1小时音频10秒转完！NVIDIA开源最强英文转写模型Canary，还能直接对话总结！(附保姆级本地部署指南)
## https://youtu.be/EFkMBkgsxl4

# 安装指南
## 创建运行环境
conda create -n canary python=3.10 -y  
conda activate canary  

## 安装 lfs
apt install git-lfs  
git lfs install  

## 克隆文件
git clone https://huggingface.co/spaces/nvidia/canary-qwen-2.5b canary  
cd canary  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 准备模型文件
huggingface-cli download nvidia/canary-qwen-2.5b --local-dir checkpoints/canary-qwen-2.5b  

## 推理
python app.py


总结一下吧，用中文回答  
英国政府为什么要掩盖真相，仅从原文中回答，用中文回答  
将转写的内容文本，全部给我翻译为中文吧  




  












 
















