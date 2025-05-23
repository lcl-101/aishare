# Youtube 节目：
## 字节跳动开源Dolphin：文档图像解析的黑科技？上手体验 (在线+本地部署)
## https://youtu.be/DSw8--IwQyc

# 安装指南

## 创建运行环境
conda create -n dolphin python=3.10 -y  
conda activate dolphin  

## 克隆项目
git clone https://github.com/ByteDance/Dolphin.git  
cd Dolphin  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 准备模型文件
huggingface-cli download ByteDance/Dolphin --local-dir checkpoints/Dolphin  

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
python app.py  
  












 
















