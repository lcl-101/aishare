# Youtube 节目：
## 告别“徒有其表”的3D空壳！NVIDIA重磅开源Part Packer：像工程师一样懂结构的3D生成神器！🔥
## https://youtu.be/2feq2YwQ8_E

# 安装指南
## 克隆项目
git clone https://github.com/NVlabs/PartPacker.git  
cd PartPacker  

## 创建运行环境
apt-get install -y build-essential cmake  
export CC="$(command -v gcc)"  
export CXX="$(command -v g++)"  
conda create -n partpacker python=3.10 -y  
conda activate partpacker  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
pip install meshiki  

## 模型下载
hf download nvidia/PartPacker --local-dir checkpoints/PartPacker  
hf download facebook/dinov2-giant --local-dir checkpoints/dinov2-giant  

## 推理演示
python app.py      

  












 
















