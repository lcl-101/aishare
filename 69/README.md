# Youtube 节目：
## Hi3DGen：2D图片极速转3D模型！精度更高，速度更快！
## https://youtu.be/GIiENsR68To

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt install git-lfs  
git lfs install  

## 克隆项目
git clone https://huggingface.co/spaces/Stable-X/Hi3DGen  
cd Hi3DGen  

## 创建和激活运行环境
conda create -n Hi3DGen python=3.10 -y  
conda activate Hi3DGen  

## 安装依赖
pip install -r requirements.txt  
pip install pydantic==2.10.6    

## 下载模型
huggingface-cli download Stable-X/trellis-normal-v0-1 --local-dir checkpoints/trellis-normal-v0-1  

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
python app.py  















