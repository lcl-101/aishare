# Youtube 节目：
## HunyuanWorld Mirror 部署教程：把你的2D图像变成逼真的3D场景 (点云/深度图/3D高斯)
## https://youtu.be/FZGDUmTTBhc

# 安装指南
## 创建运行环境
conda create -n hunyuanworld-mirror python=3.10 cmake=3.14.0 -y  
conda activate hunyuanworld-mirror   

## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git  
cd HunyuanWorld-Mirror  

## 安装依赖组件
conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia -y  
pip install -r requirements.txt  
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124  
pip install -r requirements_demo.txt      

## 模型下载
hf download tencent/HunyuanWorld-Mirror --local-dir checkpoints/HunyuanWorld-Mirror  

## 推理演示
python app.py     

  












 
















