# Youtube 节目：
## LHM：阿里开源AI，让静态人物动起来！
## https://youtu.be/Q56Jllz33tk

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
git clone https://github.com/aigc3d/LHM.git  
cd LHM  

## 创建运行环境
conda create -n LHM python=3.10 -y  
conda activate LHM  

## 安装依赖组件
sh ./install_cu121.sh  
pip install pydantic==2.10.6   

## 准备其它文件
tar -xvf LHM_prior_model.tar    
tar -xvf ./motion_video.tar   

## 推理 
python app.py --model_name LHM-1B   

## 下载动作处理模型
wget -P ./pretrained_models/human_model_files/pose_estimate https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/yolov8x.pt  
wget -P ./pretrained_models/human_model_files/pose_estimate https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/vitpose-h-wholebody.pth  

## 安装动作处理依赖
cd ./engine/pose_estimation  
pip install mmcv==1.3.9  
pip install -v -e third-party/ViTPose  
pip install ultralytics  
cd .. && cd ..  

## 动作数据提取
python ./engine/pose_estimation/video2motion.py --video_path ./train_data/demo.mp4 --output_path ./train_data/custom_data  

## 动作数据驱动图片
bash inference.sh LHM-1B  ./train_data/example_imgs/00000000_joker_2.jpg  ./train_data/custom_data/demo/smplx_params  

 
















