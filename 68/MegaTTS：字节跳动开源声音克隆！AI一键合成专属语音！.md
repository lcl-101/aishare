# Youtube 节目：
## MegaTTS：字节跳动开源声音克隆！AI一键合成专属语音！
## https://youtu.be/FjxOfQJUV1Y

# 安装指南

## 安装系统组件
sudo apt install ffmpeg  

## 克隆项目
git clone https://github.com/bytedance/MegaTTS3.git  
cd MegaTTS3  

## 创建和激活运行环境
conda create -n MegaTTS3 python=3.9 -y  
conda activate MegaTTS3  

## 安装依赖
pip install -r requirements.txt  
pip install gradio==4.44.1  
pip install pydantic==2.10.6  

## 下载模型
huggingface-cli download ByteDance/MegaTTS3 --local-dir checkpoints/  

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
export PYTHONPATH="//home/softice/MegaTTS3:$PYTHONPATH" #Linux/Mac  
CUDA_VISIBLE_DEVICES=0 python tts/gradio_api.py  












