# Youtube 节目：
## 【保姆级教程】一张图生成一个可玩的游戏世界？腾讯 Hunyuan-GameCraft 本地部署！(附独家WebUI与低配方案) 
## https://youtu.be/LKRtVtZhTVQ

# 安装指南
## 创建运行环境
conda create -n GameCraft python=3.10 -y  
conda activate GameCraft  

## 克隆文件
git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0.git  
cd Hunyuan-GameCraft-1.0  

## 安装依赖组件
pip install -r requirements.txt  
pip install torchvision==0.20.1  

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  

## 下载模型
huggingface-cli download tencent/Hunyuan-GameCraft-1.0 --local-dir checkpoints/Hunyuan-GameCraft-1.0  

## 推理
python app.py  




  












 
















