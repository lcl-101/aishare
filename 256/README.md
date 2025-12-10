# Youtube 节目：
## 腾讯Hunyuan-3D-2.1来了！零门槛、速度快、结构准，本地部署全流程教学
## https://youtu.be/8wX0l_1dHo0

# 安装指南
## 克隆项目
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git  
cd Hunyuan3D-2.1  

## 创建运行环境
conda create -n hunyuan3d-2-1 python=3.10 -y  
conda activate hunyuan3d-2-1  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
sed -i 's/^bpy==4.0$/# bpy==4.0  # 已通过 mock 方式绕过，无需安装/' requirements.txt  
pip install -r requirements.txt  
cd hy3dpaint/custom_rasterizer  
pip install --no-build-isolation -e .  
cd ../..  
cd hy3dpaint/DifferentiableRenderer  
bash compile_mesh_painter.sh  
cd ../..  

## 模型下载
hf download tencent/Hunyuan3D-2.1 --local-dir checkpoints/Hunyuan3D-2.1  
hf download facebook/dinov2-giant --local-dir checkpoints/dinov2-giant  
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt   

## 推理演示
python app.py        

  












 
















