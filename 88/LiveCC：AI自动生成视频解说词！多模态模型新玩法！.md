# Youtube 节目：
## LiveCC：AI自动生成视频解说词！多模态模型新玩法！
## https://youtu.be/4MUftZFYq38

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 安装 cuda
wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run  
sudo bash cuda_12.4.1_550.54.15_linux.run  
rm cuda_12.4.1_550.54.15_linux.run  

sudo nano ~/.bashrc  
export PATH=/usr/local/cuda-12.4/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH  
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  
source ~/.bashrc  

## 克隆项目
git clone https://github.com/showlab/livecc.git  
cd livecc  

## 创建和激活运行环境
conda create -n LiveCC python=3.11 -y    
conda activate LiveCC  

## 安装依赖组件
pip install -r requirements.txt  
pip install flash-attn --no-build-isolation  
pip install livecc-utils  

## 下载模型
huggingface-cli download chenjoya/LiveCC-7B-Instruct --local-dir checkpoints/LiveCC-7B-Instruct --exclude *README.md* *.gitattributes*  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/anaconda3/envs/LiveCC/lib/   
python demo/app.py  

## 合成效果
https://speechgen.io/en/subs/  











 
















