# Youtube 节目：
## YOLOE 开源项目分享：实时图像识别，AI提示词精准定位！
## https://youtu.be/dutEpupI0xg

# 安装指南

## 创建运行环境
conda create -n yoloe python=3.10 -y
conda activate yoloe

## 克隆项目
git clone https://github.com/THU-MIG/yoloe.git
cd yoloe

## 安装依赖组件
pip install -r requirements.txt
pip install gradio==4.42.0 gradio_image_prompter==0.1.0 fastapi==0.112.2 huggingface-hub==0.26.3 gradio_client==1.3.0

## 准备模型文件
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
pip install huggingface-hub==0.26.3
huggingface-cli download jameslahm/yoloe yoloe-v8l-seg.pt --local-dir pretrain
## 推理 
python app.py









