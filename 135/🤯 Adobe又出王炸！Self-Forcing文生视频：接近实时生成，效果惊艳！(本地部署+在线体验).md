# Youtube 节目：
## 🤯 Adobe又出王炸！Self-Forcing文生视频：接近实时生成，效果惊艳！(本地部署+在线体验)
## https://youtu.be/gGLk1KWetJg

# 安装指南
## 准备编译工具
apt update && apt upgrade -y  
apt install build-essential -y  
apt install ffmpeg -y  

## 创建运行环境
conda create -n SelfForcing python=3.10.0 -y  
conda activate SelfForcing  

## 克隆项目
git clone https://github.com/guandeh17/Self-Forcing.git SelfForcing  
cd SelfForcing  

## 安装依赖组件
pip install -r requirements.txt  
pip install flash-attn --no-build-isolation  
python setup.py develop  

## 准备模型文件
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B  
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .  

## 推理 
python demo.py  



  












 
















