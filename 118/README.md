# Youtube 节目：
## Google开源MedGemma：专为医疗打造！本地部署实测，看AI如何分析医学影像！
## https://youtu.be/kdFdV6-edIA

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 创建运行环境
conda create -n MedGemma python=3.10 -y  
conda activate MedGemma  

## 克隆项目
git clone https://github.com/Google-Health/medgemma.git  
cd medgemma  

## 安装依赖组件
pip install --upgrade transformers accelerate bitsandbytes torch pillow gradio  

## 准备模型文件
huggingface-cli download google/medgemma-4b-it --local-dir checkpoints/medgemma-4b-it  

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
python app.py  
  












 
















