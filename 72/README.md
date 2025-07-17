# Youtube 节目：
## 告别ChatGPT限流！开源AI帮你无限生成吉卜力风格图片！
## https://youtu.be/K2SLrqCUysM

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y
sudo apt install git-lfs
git lfs install

## 克隆项目
git clone https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli
cd EasyControl_Ghibli

## 创建和激活运行环境
conda create -n EasyControl python=3.10 -y
conda activate EasyControl

## 安装依赖
pip install -r requirements.txt
pip install gradio

## 推理
find . -name "*:Zone.Identifier" -type f -delete   
python app.py
















