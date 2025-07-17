# Youtube 节目：
## 字节跳动Bagel模型本地部署实战：文生图、AI P图、看图说话，秀翻全场！
## https://youtu.be/9vFt_xdE3U0

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 安装 cuda
wget http://cache.mixazure.com:88/cuda/cuda_12.4.1_550.54.15_linux.run  
sudo bash cuda_12.4.1_550.54.15_linux.run  
rm cuda_12.4.1_550.54.15_linux.run  

sudo nano ~/.bashrc  
export PATH=/usr/local/cuda-12.4/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH  
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  
source ~/.bashrc  

## 创建运行环境
conda create -n bagel python=3.10 -y  
conda activate bagel  

## 克隆项目
git clone https://github.com/bytedance-seed/BAGEL.git  
cd BAGEL  

## 安装依赖组件
pip install torch==2.5.1 torchvision==0.20.1  
pip install -r requirements.txt  
pip install gradio  

## 准备模型文件
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir checkpoints/BAGEL-7B-MoT  

## 推理 
find . -name "*:Zone.Identifier" -type f -delete    
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6    
python app.py  
  












 
















