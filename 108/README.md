# Youtube 节目：
## 腾讯开源NormalCrafter：从视频生成高质量法线序列 (3D重建利器)
## https://youtu.be/2zMvMozm5bM

# 安装指南

## 安装 lfs 组件
sudo apt install ffmpeg -y  
sudo apt-get install git-lfs  
git lfs install  

## 克隆项目
git clone https://github.com/Binyr/NormalCrafter.git  
cd NormalCrafter  

## 创建和激活运行环境
conda create -n NormalCrafter python=3.10 -y  
conda activate NormalCrafter  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型
huggingface-cli download Yanrui95/NormalCrafter --local-dir checkpoints/NormalCrafter  
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir checkpoints/stable-video-diffusion-img2vid-xt  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete    
python app.py      











 
















