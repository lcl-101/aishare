# Youtube 节目：
## Step1X-3D 开源框架，图片秒变高保真模型，效果直逼商业级！
## https://youtu.be/AJ0vDhvhdYQ

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y         
sudo apt install git-lfs -y      
git lfs install       
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
conda create -n 3D python=3.10 -y    
conda activate 3D  

## 克隆项目
git clone https://huggingface.co/spaces/stepfun-ai/Step1X-3D  
cd Step1X-3D  

## 安装依赖组件
pip install -r requirements.txt  

cd step1x3d_texture/differentiable_renderer/ && python setup.py install  
cd .. && cd ..  
pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl  
pip install pydantic==2.10.6   

## 下载模型文件
pip install "huggingface_hub[cli]"  
huggingface-cli download madebyollin/sdxl-vae-fp16-fix --local-dir checkpoints/sdxl-vae-fp16-fix  
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir checkpoints/stable-diffusion-xl-base-1.0  
huggingface-cli download stepfun-ai/Step1X-3D --local-dir checkpoints/Step1X-3D  

## 修改程序
/home/softice/Step1X-3D/step1x3d_texture/pipelines/step1x_3d_texture_synthesis_pipeline.py  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6   
python app.py  
  












 
















