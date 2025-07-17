# Youtube 节目：
## STDGen：开源AI 2D转3D新方案，多视角生成+Blender编辑
## https://youtu.be/vxFc2bNmcB8

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt-get install git-lfs   
sudo apt install build-essential -y

## 安装 cuda
wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run  
sudo bash cuda_12.4.1_550.54.15_linux.run  
rm cuda_12.4.1_550.54.15_linux.run

## 创建运行环境
conda create -n stdgen python=3.9 -y  
conda activate stdgen

## 克隆项目
git clone https://huggingface.co/spaces/hyz317/StdGEN  
cd StdGEN

## 安装依赖组件
pip install -r pre-requirements.txt  
pip install -r requirements.txt  
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0%2Bcu124.html

## 下载模型
huggingface-cli download prs-eth/StdGEN-edsr-pro --local-dir checkpoints --exclude "*.git*" "README.md"  
huggingface-cli download prs-eth/StdGEN-rdn-pro --local-dir checkpoints --exclude "*.git*" "README.md"

## 启动 WebUI
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/anaconda3/envs/stdgen/lib/  
python app.py










