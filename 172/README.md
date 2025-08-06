# Youtube 节目：
## AI一键抠图神器 RemBG：官方模型效果不佳？教你从零训练专属模型，精准去除任何背景！
## https://youtu.be/pFBeh0iR8Ds

# 安装指南
## 创建运行环境
conda create -n rembg python=3.10 -y  
conda activate rembg  

## 创建目录
mkdir rembg  
cd rembg  

## 安装组件
pip install "rembg[gpu,cli]" # for library + cli  

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" && echo "已设置 LD_LIBRARY_PATH: $LD_LIBRARY_PATH"  

## 准备模型文件
mkdir models  
cd models  
wget http://cache.mixazure.com/whl/u2net.onnx  
wget http://cache.mixazure.com/whl/u2netp.onnx  
wget http://cache.mixazure.com/whl/isnet-general-use.onnx  
wget http://cache.mixazure.com/whl/isnet-anime.onnx  
wget http://cache.mixazure.com/whl/custom.onnx  

## 推理
python app.py  


## 训练
git clone https://github.com/xuebinqin/U-2-Net.git  
cd U-2-Net/  
pip install numpy scikit-image pillow opencv-python matplotlib onnx  
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  

wget http://cache.mixazure.com/whl/data.zip  
unzip data.zip  

cd saved_models/  
wget http://cache.mixazure.com/whl/latest.pth  
cd ..   



  












 
















