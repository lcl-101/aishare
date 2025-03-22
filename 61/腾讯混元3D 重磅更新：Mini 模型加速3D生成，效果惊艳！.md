# Youtube 节目：
## 腾讯混元3D 重磅更新：Mini 模型加速3D生成，效果惊艳！
## https://youtu.be/jRO_2mE7d24

# 安装指南

## 安装编译工具和 cuda
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential -y
wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
rm cuda_12.4.1_550.54.15_linux.run

## 添加环境变量
sudo nano ~/.bashrc

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

source ~/.bashrc

sudo nano /etc/ld.so.conf

/usr/local/cuda-12.4/lib64
sudo ldconfig

## 创建运行环境
conda create -n Hunyuan3D python=3.10 -y
conda activate Hunyuan3D

## 克隆项目
git clone https://github.com/Tencent/Hunyuan3D-2.git
cd Hunyuan3D-2

## 安装依赖
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install SentencePiece

## for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
cd ../../..

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/anaconda3/envs/Hunyuan3D/lib/

修改 gradio_app.py 注释 717 行
修改 hy3dgen\shapegen\utils.py 注释 97 行

### Hunyuan3D-2mini
python3 gradio_app.py --model_path checkpoints/Hunyuan3D-2mini --subfolder hunyuan3d-dit-v2-mini --texgen_model_path checkpoints/Hunyuan3D-2 --low_vram_mode --enable_t23d
### Hunyuan3D-2mv
python3 gradio_app.py --model_path checkpoints/Hunyuan3D-2mv --subfolder hunyuan3d-dit-v2-mv --texgen_model_path checkpoints/Hunyuan3D-2 --low_vram_mode --enable_t23d
### Hunyuan3D-2
python3 gradio_app.py --model_path checkpoints/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0 --texgen_model_path checkpoints/Hunyuan3D-2 --low_vram_mode --enable_t23d











