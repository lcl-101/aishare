# Youtube 节目：
## 40倍推理加速！普通显卡也能跑SOTA级文生图？LightX2V 零日适配 Qwen-Image-2512 实战部署 🔥
## https://youtu.be/Xk6KvgVvrps

# 安装指南
## 克隆项目
git clone https://github.com/ModelTC/LightX2V.git  
cd LightX2V  

## 创建运行环境
conda create -n lightx python=3.10 -y  
conda activate lightx   

## 安装依赖组件
pip install -v .  

## 模型下载
hf download Qwen/Qwen-Image-2512 --local-dir checkpoints/Qwen-Image-2512  
hf download lightx2v/Qwen-Image-2512-Lightning --local-dir checkpoints/Qwen-Image-2512-Lightning  

## 推理演示
python app.py        

  












 
















