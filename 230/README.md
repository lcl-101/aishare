# Youtube 节目：
## AI 配音导演 MAYA1：一句话指令，让AI“演”出喜怒哀乐！从零开始本地部署+实战演示。
## https://youtu.be/1OJfSWwOpm4

# 安装指南
## 克隆项目
mkdir maya1  
cd maya1  

## 创建运行环境
conda create -n maya1 python=3.10 -y  
conda activate maya1   

## 安装依赖组件
pip install torch transformers snac soundfile accelerate gradio   

## 模型下载
hf download maya-research/maya1 --local-dir checkpoints/maya1  
hf download hubertsiuzdak/snac_24khz --local-dir checkpoints/snac_24khz  

## 推理演示
python app.py  

  












 
















