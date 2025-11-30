# Youtube 节目：
## Flux1 只是开始？Flux2 评测与部署：32B 参数的“绘图航母”到底有多强？
## https://youtu.be/BydoHYvRoUc

# 安装指南
## 创建项目目录
mkdir flux2  
cd flux2  

## 创建运行环境
conda create -n flux2 python=3.10 -y    
conda activate flux2    

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/diffusers.git  
pip install --upgrade transformers accelerate bitsandbytes  
pip install gradio  

## 模型下载
hf download black-forest-labs/FLUX.2-dev --local-dir checkpoints/FLUX.2-dev  

## 推理演示
python app.py      

  












 
















