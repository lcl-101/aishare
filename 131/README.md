# Youtube 节目：
## NVIDIA 开源神器 PartPacker！单张图片秒生3D模型，还能智能拆分部件！【在线+本地部署教程】
## https://youtu.be/RjDc6euzQt0

# 安装指南
## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 克隆项目
git clone https://github.com/NVlabs/PartPacker.git  
cd PartPacker  

## 创建和激活运行环境
conda create -n PartPacker python=3.10 -y  
conda activate PartPacker  

## 安装依赖组件
http://cache.mixazure.com:88/whl/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl  
pip install -r requirements.txt  
pip install transformers  
pip install python-fcl  

## 下载模型文件
mkdir pretrained  
cd pretrained  
wget https://huggingface.co/nvidia/PartPacker/resolve/main/vae.pt  
wget https://huggingface.co/nvidia/PartPacker/resolve/main/flow.pt  

## 启动程序
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6    
python app.py  



  












 
















