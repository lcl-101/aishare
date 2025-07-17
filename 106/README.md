# Youtube 节目：
## 腾讯AI新作Primitive Anything：3D模型秒变“积木”，快速原型不是梦！
## https://youtu.be/Sd7j5gvX9NQ

# 安装指南

## 配置安全组
sudo usermod -aG video $USER  
sudo usermod -aG render $USER  

## 安装系统组件
sudo apt update && sudo apt upgrade -y    
sudo apt install build-essential -y    
sudo apt install git-lfs -y    
git lfs install     

## 克隆项目
git clone https://huggingface.co/spaces/hyz317/PrimitiveAnything  
cd PrimitiveAnything  

## 创建和激活运行环境
conda create -n PrimitiveAnything python=3.10 -y  
conda activate PrimitiveAnything  

## 安装依赖组件
pip install -r pre-requirements.txt  
pip install -r requirements.txt  

## 下载模型
huggingface-cli download hyz317/PrimitiveAnything --local-dir ckpt/  
wget -P ckpt/ https://huggingface.co/Maikou/Michelangelo/resolve/main/checkpoints/aligned_shape_latents/shapevae-256.ckpt  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete    
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6    
python app.py    










 
















