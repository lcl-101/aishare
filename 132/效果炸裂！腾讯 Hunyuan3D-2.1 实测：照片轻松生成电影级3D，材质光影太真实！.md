# Youtube 节目：
## 效果炸裂！腾讯 Hunyuan3D-2.1 实测：照片轻松生成电影级3D，材质光影太真实！
## https://youtu.be/6OP7jmMvOqY

# 安装指南
## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  
sudo apt-get install libsm6 libxrender1 libfontconfig1  

## 安装 cuda
wget http://cache.mixazure.com:88/cuda/cuda_12.4.1_550.54.15_linux.run  
sudo bash cuda_12.4.1_550.54.15_linux.run  
rm cuda_12.4.1_550.54.15_linux.run  

sudo nano ~/.bashrc  
export PATH=/usr/local/cuda-12.4/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH  
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  
source ~/.bashrc  

## 克隆项目
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git Hunyuan3D  
cd Hunyuan3D  

## 创建和激活运行环境
conda create -n Hunyuan3D python=3.10 -y  
conda activate Hunyuan3D  

## 安装依赖组件
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  

cd hy3dpaint/custom_rasterizer  
pip install -e .  
cd ../..  
cd hy3dpaint/DifferentiableRenderer  
bash compile_mesh_painter.sh  
cd ../..  

## 启动程序
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6    
python3 gradio_app.py   --model_path tencent/Hunyuan3D-2.1   --subfolder hunyuan3d-dit-v2-1   --texgen_model_path tencent/Hunyuan3D-2.1   --low_vram_mode  

\hy3dpaint\ckpt  
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth  



  












 
















