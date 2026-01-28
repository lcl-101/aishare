# Youtube 节目：
## AI绘图界新黑马？Z-Image-Turbo 保姆级部署教程：消费级显卡也能生成的顶尖画质！
## https://youtu.be/Fh0xst4mF5Q

# 安装指南
## 克隆项目
git clone https://github.com/Tongyi-MAI/Z-Image.git  
cd Z-Image    

## 创建运行环境
conda create -n z-image python=3.10 -y  
conda activate z-image   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/diffusers  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
pip install gradio  

## 模型下载
hf download Tongyi-MAI/Z-Image-Turbo --local-dir checkpoints/Z-Image-Turbo    

## 推理演示
python app.py      

  












 
















