# Youtube 节目：
## 🤯 照片「活」了！腾讯开源 AI 神器 HunyuanPortrait，一键让静态肖像动起来 (本地部署+WebUI实测)
## https://youtu.be/l--cMnnL3Rs

# 安装指南
## 安装系统组件
sudo apt update && sudo apt upgrade -y         
sudo apt install ffmpeg -y   

## 创建和激活运行环境
conda create -n HunyuanPortrait python=3.10 -y    
conda activate HunyuanPortrait  

## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HunyuanPortrait  
cd HunyuanPortrait  

## 安装依赖组件
pip install torch torchvision torchaudio  
pip install -r requirements.txt  
pip install gradio  

## 下载模型文件
pip3 install "huggingface_hub[cli]"  
mkdir pretrained_weights  
cd pretrained_weights  
huggingface-cli download --resume-download stabilityai/stable-video-diffusion-img2vid-xt --local-dir . --include "*.json"  
wget -c https://huggingface.co/LeonJoe13/Sonic/resolve/main/yoloface_v5m.pt  
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors -P vae  
wget -c https://huggingface.co/FoivosPar/Arc2Face/resolve/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx  
huggingface-cli download --resume-download tencent/HunyuanPortrait --local-dir hyportrait  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py    
  












 
















