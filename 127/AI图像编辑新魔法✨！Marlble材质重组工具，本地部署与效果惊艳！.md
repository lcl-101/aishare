# Youtube 节目：
## AI图像编辑新魔法✨！Marlble材质重组工具，本地部署与效果惊艳！
## https://youtu.be/RAd6n9Oznbc

# 安装指南
## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 克隆项目
git clone https://github.com/Stability-AI/marble.git  
cd marble  

## 创建和激活运行环境
conda create -n marble python=3.9.7 -y  
conda activate marble  

## 安装依赖组件
pip install -r requirements.txt  
pip install numpy==1.26.4  
pip install pydantic==2.10.6    

## 启动程序 修改地址
find . -name "*:Zone.Identifier" -type f -delete  
python gradio_demo.py  
  












 
















