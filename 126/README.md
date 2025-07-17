# Youtube 节目：
## 【AI尝鲜】VUI项目实测：两张4090训练，本地就能跑的对话式AI！
## https://youtu.be/c0Hhg54Hr0o

# 安装指南
## 安装系统组件
sudo apt update && sudo apt upgrade -y   
sudo apt install build-essential -y  

## 克隆项目
git clone https://github.com/fluxions-ai/vui.git 
cd vui  

## 创建和激活运行环境
conda create -n vui python=3.12 -y  
conda activate vui  

## 安装依赖组件
pip install -e .  

## 下载模型文件
huggingface-cli login  

## 启动程序
python demo.py  
  












 
















