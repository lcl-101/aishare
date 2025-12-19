# Youtube 节目：
## Trellis2 震撼开源：40亿参数挑战商业级 3D 生成，PBR材质/透明纹理完美还原！🔥 本地部署保姆级教程
## https://youtu.be/t4uRB7OA7ko

# 安装指南
## 克隆项目
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive  
cd TRELLIS.2  

## 创建运行环境
conda create -n trellis2 python=3.10 -y  
conda activate trellis2   

## 安装依赖组件
### 1. 安装 PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124  
### 2. 安装 flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation  
### 3. 安装基础依赖
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio tensorboard pandas lpips zstandard  
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8  
pip install kornia timm  
### 4. 安装 Pillow（带 WEBP 支持）
apt install -y libjpeg-dev libwebp-dev  
pip install pillow --no-cache-dir  
### 5. 准备扩展目录并克隆仓库
mkdir -p /tmp/extensions  
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast  
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec  
git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive  
cp -r o-voxel /tmp/extensions/o-voxel  
### 6. 单独处理 CuMesh（修复 SSH submodule 问题）
git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh  
cd /tmp/extensions/CuMesh  
git config submodule.third_party/cubvh.url https://github.com/JeffreyXiang/cubvh.git  
git submodule update --init --recursive  
cd /workspace/TRELLIS.2  
### 7. 修改 o-voxel 的依赖配置，避免重复从 git 安装
sed -i 's|"cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git",|"cumesh",|' /tmp/extensions/o-voxel/pyproject.toml  
sed -i 's|"flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git",|"flex_gemm",|' /tmp/extensions/o-voxel/pyproject.toml  
### 8. 按顺序安装所有扩展
pip install /tmp/extensions/nvdiffrast --no-build-isolation  
pip install /tmp/extensions/nvdiffrec --no-build-isolation  
pip install /tmp/extensions/CuMesh --no-build-isolation  
pip install /tmp/extensions/FlexGEMM --no-build-isolation  
pip install /tmp/extensions/o-voxel --no-build-isolation  

## 模型下载
hf download microsoft/TRELLIS.2-4B --local-dir checkpoints/TRELLIS.2-4B  

## 推理演示
python app.py        

  












 
















