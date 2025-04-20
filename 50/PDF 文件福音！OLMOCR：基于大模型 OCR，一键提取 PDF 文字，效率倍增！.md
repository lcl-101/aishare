# Youtube 节目：
## PDF 文件福音！OLMOCR：基于大模型 OCR，一键提取 PDF 文字，效率倍增！
## https://youtu.be/HRmkCSgqU3o

# 安装指南

## 创建运行环境
conda create -n olmocr -y python=3.11  
conda activate olmocr  

## 克隆项目
mkdir olmocr  
cd olmocr  

## 安装依赖的组件
sudo apt-get install poppler-utils  
pip install olmocr  
pip install pypdf2  
pip install gradio  

## 下载模型文件
huggingface-cli download allenai/olmOCR-7B-0225-preview --local-dir checkpoints/olmOCR-7B-0225-preview  
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir checkpoints/Qwen2-VL-7B-Instruct  

## 运行 webui 程序
python app.py  








