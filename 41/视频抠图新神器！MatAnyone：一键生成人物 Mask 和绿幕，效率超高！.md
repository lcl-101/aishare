# Youtube 节目：
## 视频抠图新神器！MatAnyone：一键生成人物 Mask 和绿幕，效率超高！
## https://youtu.be/XLUhp2lhNgs

# 安装指南



## 安装 lfs 和 ffmpeg
sudo apt-get update  
sudo apt-get install git  
sudo apt-get install git-lfs  
sudo apt install ffmpeg -y  


## 克隆项目
git lfs install  
git clone https://huggingface.co/spaces/PeiqingYang/MatAnyone  
cd MatAnyone/  
## 创建运行环境
conda create -n MatAnyone python=3.10.0 -y    
conda activate MatAnyone  

## 安装编译组件
sudo apt install build-essential -y  
## 安装依赖组件
pip install -r requirements.txt  

## 启动 webui
cd matanyone/  
python app.py  







