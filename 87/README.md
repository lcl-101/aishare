# Youtube 节目：
## Describe Anything：自动分析视频、图片，一键生成描述！
## https://youtu.be/ryPVIaD-GKE

# 安装指南

## 克隆项目
git clone https://github.com/NVlabs/describe-anything  
cd describe-anything  

## 创建和激活运行环境
conda create -n Da python=3.10 -y  
conda activate Da  

## 安装依赖组件
pip install -v .  
pip install sam2  

## 下载模型
huggingface-cli download facebook/sam-vit-huge --local-dir checkpoints/sam-vit-huge  
huggingface-cli download nvidia/DAM-3B --local-dir checkpoints/DAM-3B  
huggingface-cli download nvidia/DAM-3B-Video --local-dir checkpoints/DAM-3B-Video  
huggingface-cli download facebook/sam2.1-hiera-large --local-dir checkpoints/sam2.1-hiera-large  


## 启动程序
find . -name "*:Zone.Identifier" -type f -delete    
python demo_simple.py  

mkdir -p configs/sam2.1  
mv checkpoints/sam2.1-hiera-large/sam2.1_hiera_l.yaml configs/sam2.1/  
python demo_video.py  










 
















