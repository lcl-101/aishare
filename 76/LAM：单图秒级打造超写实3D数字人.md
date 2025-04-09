# Youtube 节目：
## LAM：单图秒级打造超写实3D数字人
## https://youtu.be/pfF3D9hH16M

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install gcc-11 g++-11 -y  
sudo ln -s /usr/bin/gcc-11 /usr/local/bin/gcc  
sudo ln -s /usr/bin/g++-11 /usr/local/bin/g++  
sudo ln -s /usr/bin/g++-11 /usr/local/bin/c++  

## 安装 cuda
wget http://cache.mixazure.com/cuda/cuda_12.1.0_530.30.02_linux.run  
sudo bash cuda_12.1.0_530.30.02_linux.run  
rm cuda_12.1.0_530.30.02_linux.run  

sudo nano ~/.bashrc  
export PATH=/usr/local/cuda-12.1/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH  
source ~/.bashrc  

## 克隆项目
git clone https://github.com/aigc3d/LAM.git  
cd LAM  

## 创建运行环境
conda create -n LAM python=3.10 -y  
conda activate LAM  

## 安装依赖组件
sh  ./scripts/install/install_cu121.sh  
pip install git+https://github.com/NVlabs/nvdiffrast.git  
pip install gradio==3.44.3  

## 准备测试文件和模型文件
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_assets.tar  
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_human_model.tar  
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/tracking_pretrain_model.tar  
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_20K.tar  

find . -name "*:Zone.Identifier" -type f -delete  
tar -xf LAM_assets.tar && rm LAM_assets.tar  
tar -xf LAM_human_model.tar && rm LAM_human_model.tar  
tar -xf tracking_pretrain_model.tar && rm tracking_pretrain_model.tar  
tar -xf LAM_20K.tar && rm LAM_20K.tar  

## 推理 
python app_lam.py  


 
















