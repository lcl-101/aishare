# Youtube 节目：
## 效果炸裂，但4秒视频要跑15分钟？阿里最强数字人项目Fantasy-Portrait本地部署实测！  
## https://youtu.be/iFr8P54eopA  

# 安装指南
## 创建运行环境
conda create -n FantasyPortrait python=3.10 pip=22.3 -y  
conda activate FantasyPortrait  

## 克隆文件
git clone https://github.com/Fantasy-AMAP/fantasy-portrait.git fantasyportrait  
cd fantasyportrait  

## 安装依赖组件
pip install -r requirements.txt  
pip install onnx onnxruntime gradio  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl   

## 下载模型
pip install "huggingface_hub[cli]"  
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir checkpoints/Wan2.1-I2V-14B-720P  
huggingface-cli download acvlab/FantasyPortrait --local-dir checkpoints/FantasyPortrait  

## 推理
python app.py  




  












 
















