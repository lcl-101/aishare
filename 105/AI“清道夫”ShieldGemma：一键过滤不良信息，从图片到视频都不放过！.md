# Youtube 节目：
## AI“清道夫”ShieldGemma：一键过滤不良信息，从图片到视频都不放过！
## https://youtu.be/XNKt1oIpA6I

# 安装指南

## 创建项目目录
mkdir ShieldGemma2  
cd ShieldGemma2  

## 创建和激活运行环境
conda create -n ShieldGemma2 python=3.10 -y  
conda activate ShieldGemma2  

## 安装依赖组件
pip install torch transformers gradio Pillow opencv-python  

## 下载模型
pip install "huggingface_hub[cli]"  
huggingface-cli download google/shieldgemma-2-4b-it --local-dir checkpoints/shieldgemma-2-4b-it  
 
## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py  









 
















