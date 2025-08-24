# Youtube 节目：
## 手机厂商的抠图黑科技？vivo 开源 SDMatte 本地部署，标准版 vs 轻量版效果大比拼！
## https://youtu.be/y386e7GdqJs  

# 安装指南
## 创建和激活运行环境
conda create -n SDMatte python==3.10 -y  
conda activate SDMatte  

## 克隆项目
git clone https://github.com/vivoCameraResearch/SDMatte.git  
cd SDMatte  

## 安装依赖组件
pip install -r requirements.txt  
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'  
pip install gradio  

## 下载模型文件
huggingface-cli download LongfeiHuang/SDMatte --local-dir checkpoints/SDMatte  
huggingface-cli download LongfeiHuang/LiteSDMatte --local-dir checkpoints/LiteSDMatte  

## 启动程序
python app.py  




  












 
















