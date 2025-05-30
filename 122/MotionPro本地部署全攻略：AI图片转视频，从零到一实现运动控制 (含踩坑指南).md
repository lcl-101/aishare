# Youtube 节目：
## MotionPro本地部署全攻略：AI图片转视频，从零到一实现运动控制 (含踩坑指南)
## https://youtu.be/wbIt1HDKRSI

# 安装指南
## 安装系统组件
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

## 创建和激活运行环境
conda create -n motionpro python=3.10.0 -y  
conda activate motionpro  

## 克隆项目
git clone https://github.com/HiDream-ai/MotionPro.git  
cd MotionPro  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型文件
huggingface-cli download HiDream-ai/MotionPro --local-dir checkpoints/  

## 修改程序
/home/softice/MotionPro/tools/ZoeDepth/zoedepth/models/base_models/midas.py  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete   
python demo_sparse_flex_wh.py  
python demo_sparse_flex_wh_pure_camera.py.py    
  












 
















