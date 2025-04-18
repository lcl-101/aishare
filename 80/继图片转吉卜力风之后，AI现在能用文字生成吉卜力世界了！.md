# Youtube 节目：
## HighDream：开源AI绘画，媲美Flux模型的高质量图片生成！
## https://youtu.be/NLjM5Mb-17E

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 创建目录
mkdir HiDream  
cd HiDream  

## 创建和激活运行环境
conda create -n HiDream python=3.10 -y    
conda activate HiDream  

## 安装依赖组件
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/diffusers.git  
pip install transformers bitsandbytes accelerate pillow optimum auto-gptq gradio  

## 下载模型
huggingface-cli download azaneko/HiDream-I1-Full-nf4 --local-dir checkpoints/HiDream-I1-Full-nf4  
huggingface-cli download azaneko/HiDream-I1-Dev-nf4 --local-dir checkpoints/HiDream-I1-Dev-nf4  
huggingface-cli download azaneko/HiDream-I1-Fast-nf4 --local-dir checkpoints/HiDream-I1-Fast-nf4  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete    
python app.py  

Create an illustration of a steampunk witch, a clockwork raven, and alchemy tattoos.  

Glowing Ragdoll, USB cord nest, Keyboard hills  




 
















