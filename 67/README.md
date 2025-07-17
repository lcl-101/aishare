# Youtube 节目：
## 阿里最新开源Qwen Omni：多模态理解，AI秒懂视频、图片、音频！
## https://youtu.be/AtZJEQ9enmU

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y   
sudo apt install build-essential -y  
sudo apt install ffmpeg -y  

## 安装 CUDA 组件
wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run    
sudo sh cuda_12.4.1_550.54.15_linux.run    

## 添加环境变量
rm cuda_12.4.1_550.54.15_linux.run    
sudo nano ~/.bashrc    

export PATH=/usr/local/cuda-12.4/bin:$PATH    
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH    

source ~/.bashrc    


## 克隆项目
git clone https://github.com/QwenLM/Qwen2.5-Omni.git  
cd Qwen2.5-Omni  

## 创建和激活运行环境
conda create -n QwenOmni python=3.10 -y  
conda activate QwenOmni  

## 安装依赖
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements_web_demo.txt  
pip install qwen-omni-utils[decord]  
pip install pydantic==2.10.6  

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
python web_demo.py   












