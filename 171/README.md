# Youtube 节目：
## 海报神器！阿里通义千问Qwen-Image本地部署，图片文字精准生成！(附独家WebUI/60GB+显存预警)
## https://youtu.be/z9aHlzbD598

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

## 推理
python app.py  



  












 
















