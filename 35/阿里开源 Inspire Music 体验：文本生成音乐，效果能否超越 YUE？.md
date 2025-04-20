# Youtube 节目：
## 阿里开源 Inspire Music 体验：文本生成音乐，效果能否超越 YUE？
## https://youtu.be/9CP32UeVtkg

# 安装指南

## 克隆项目 
git clone --recursive https://github.com/FunAudioLLM/InspireMusic.git  
cd InspireMusic  


## 创建运行环境
conda create -n inspiremusic python=3.10 -y  
conda activate inspiremusic  
cd InspireMusic  
conda install -y -c conda-forge pynini==2.1.5  
pip install -r requirements.txt  
pip install flash-attn --no-build-isolation  

## 安装其它组件
sudo apt-get install sox libsox-dev -y  
sudo apt-get install ffmpeg -y  

## 准备模型文件
sudo apt-get install git-lfs  
git lfs install  
mkdir -p pretrained_models  
git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long pretrained_models/InspireMusic-1.5B-Long  

## 推理
python app.py  







