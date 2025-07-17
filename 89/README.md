# Youtube 节目：
## Phantom：图片+提示词，让AI视频更可控，保持一致性！
## https://youtu.be/cmCSj0DcrFI

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
git clone https://github.com/Phantom-video/Phantom  
cd Phantom  

## 创建和激活运行环境
conda create -n Phantom python=3.10 -y    
conda activate Phantom  

## 安装依赖组件
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install gradio  

## 下载模型
pip install "huggingface_hub[cli]"  
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B  
huggingface-cli download bytedance-research/Phantom --local-dir ./Phantom-Wan-1.3B  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py  










 
















