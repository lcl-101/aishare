# Youtube 节目：
## 【AI抠图神器】告别PS！Rembg本地部署，一键移除任何背景，还能训练自己的模型！
## https://youtu.be/6br4yo7va9w

# 安装指南
## 克隆项目
mkdir rembg  
cd rembg  
git clone https://github.com/xuebinqin/U-2-Net.git  

## 创建运行环境
conda create -n rembg python=3.10 -y  
conda activate rembg    

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install numpy scikit-image pillow opencv-python matplotlib onnx  
pip install "rembg[gpu,cli]"  

## 模型下载
mkdir models  
cd models  
wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx  

## 推理演示
python app.py     

  












 
















