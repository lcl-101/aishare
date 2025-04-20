# Youtube 节目：
## 让照片动起来！AI 项目 ViewCrafter：普通照片秒变 360° 全景视频！
## https://youtu.be/WhfI13dnXII

# 安装指南
## 安装 cuda 
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run  
sudo bash cuda_12.4.1_550.54.15_linux.run  
rm cuda_12.4.1_550.54.15_linux.run  

sudo nano ~/.bashrc  
export PATH=/usr/local/cuda-12.4/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH  
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  
source ~/.bashrc  

## 克隆代码
cd /home/softice  
git clone https://github.com/Drexubery/ViewCrafter.git  
cd ViewCrafter  

## 创建运行环境
conda create -n viewcrafter python=3.9.16 -y  
conda activate viewcrafter  
pip install -r requirements.txt  

## 安装 PyTorch3D
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2  

## 下载模型
mkdir -p checkpoints/  
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/  
wget https://huggingface.co/Drexubery/ViewCrafter_25/resolve/main/model.ckpt -P checkpoints/  
wget https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt -P checkpoints/  

## 单图推理
python gradio_app.py  
## 多图推理
sh run_sparse.sh  




