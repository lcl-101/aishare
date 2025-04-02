# Youtube 节目：
## TripoSG 专业级AI 2D转3D？Tripo3D：知名公司开源，效果惊艳！
## https://youtu.be/eWa4TV3JglA

# TripoSG 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt install git-lfs  
git lfs install  

## 克隆项目
git clone https://huggingface.co/spaces/VAST-AI/TripoSG  
cd TripoSG  

## 创建和激活运行环境
conda create -n TripoSG python=3.10 -y  
conda activate TripoSG  

## 安装依赖
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install gradio  

## 下载模型
huggingface-cli download VAST-AI/TripoSG --local-dir checkpoints/TripoSG  
huggingface-cli download briaai/RMBG-1.4 --local-dir checkpoints/RMBG-1.4  

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/anaconda3/envs/TripoSG/lib/   
python app.py  

# MV-Adapter-Img2Texture 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt install git-lfs  
git lfs install  

## 克隆项目
git clone https://huggingface.co/spaces/VAST-AI/MV-Adapter-Img2Texture  
cd MV-Adapter-Img2Texture  

## 创建和激活运行环境
conda create -n MVAdapter python=3.10 -y  
conda activate MVAdapter  

## 安装依赖
pip install -r requirements.txt  
pip install gradio  

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
python app.py  
















