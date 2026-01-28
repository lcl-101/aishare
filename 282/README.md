# Youtube 节目：
## 免费商用！3秒克隆声音的开源神器 Chatterbox：支持笑声/叹气情感控制，本地部署保姆级教程
## https://youtu.be/11tkYyWW2K8

# 安装指南
## 克隆项目
git clone https://github.com/resemble-ai/chatterbox.git  
cd chatterbox  

## 创建运行环境
conda create -n chatterbox python=3.10 -y  
conda activate chatterbox  

## 安装依赖组件
pip install -e . 

## 模型下载
hf download ResembleAI/chatterbox --local-dir checkpoints/chatterbox  
hf download ResembleAI/chatterbox-turbo --local-dir checkpoints/chatterbox-turbo  

## 推理演示
python app.py        

  












 
















