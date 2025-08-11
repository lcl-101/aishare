# Youtube 节目：
## Qwen-Image 速度革命！8步推理媲美50步，出图速度暴涨10倍！Qwen-Image-Lighting LoRA 保姆级部署教程
## https://youtu.be/XEVg16tsybE

# 安装指南
## 创建运行环境
conda create -n QwenImage python=3.10 -y  
conda activate QwenImage  

## 克隆文件
git clone https://github.com/QwenLM/Qwen-Image  
cd Qwen-Image  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers  
pip install git+https://github.com/huggingface/diffusers  
pip install accelerate gradio  

## 下载模型
huggingface-cli download Qwen/Qwen-Image --local-dir checkpoints/Qwen-Image   
huggingface-cli download lightx2v/Qwen-Image-Lightning --local-dir checkpoints/Qwen-Image-Lightning  

huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI --local-dir checkpoints/Qwen-Image_ComfyUI  

## 推理
python app.py    





  












 
















