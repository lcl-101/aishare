# Youtube 节目：
## Make-It-Animatable：让3D模型动起来！AI一键生成3D角色动画！
## https://youtu.be/xv81fbGMaVE

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt-get install git-lfs  
sudo apt install libsm6 -y  

## 克隆项目
git clone https://github.com/jasongzy/Make-It-Animatable --recursive --single-branch  
cd Make-It-Animatable  

## 创建和激活运行环境
conda create -n mia-demo python=3.11 -y  
conda activate mia-demo  

## 安装依赖
pip install -r requirements-demo.txt  

## 准备数据
GIT_LFS_SKIP_SMUDGE=1 git -C data clone https://huggingface.co/datasets/jasongzy/Mixamo  
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/jasongzy/Make-It-Animatable /tmp/hf-data  
git -C data/Mixamo lfs pull -I 'bones*.fbx,animation'  

## 下载模型
git -C /tmp/hf-data lfs pull -I output/best/new  
mkdir -p output/best && cp -r /tmp/hf-data/output/best/new output/best/  

## 下载测试数据
git -C /tmp/hf-data lfs pull -I data  
cp -r /tmp/hf-data/data/* data/  

## 下载工具
wget https://github.com/facebookincubator/FBX2glTF/releases/download/v0.9.7/FBX2glTF-linux-x64 -O util/FBX2glTF  
chmod +x util/FBX2glTF  

## 推理
python app.py  


 
















