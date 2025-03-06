# Youtube 节目：
## 图片无损放大！MoD：AI 高清扩图，告别模糊，细节更清晰！
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








