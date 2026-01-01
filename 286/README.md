# Youtube 节目：
## 告别“AI脸”与文字乱码！阿里Qwen-Image-2512炸裂开源，本地部署全攻略 🔥
## https://youtu.be/Da3wYK5uVYw

# 安装指南
## 克隆项目
mkdir qwenimage  
cd qwenimage  

## 创建运行环境
conda create -n qwenimage python=3.10 -y  
conda activate qwenimage  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/diffusers  
pip install transformers gradio accelerate  

## 模型下载
hf download Qwen/Qwen-Image-2512 --local-dir checkpoints/Qwen-Image-2512  

## 推理演示
python app.py        

  












 
















