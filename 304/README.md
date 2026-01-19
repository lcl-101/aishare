# Youtube 节目：
## 给AI装上“嘴”和“心”！NVIDIA PersonaPlex 深度体验：支持随时打断、音色克隆的实时对话神器
## https://youtu.be/G8sMnKMaDRQ

# 安装指南
## 克隆项目
git clone https://github.com/NVIDIA/personaplex.git  
cd personaplex  

## 创建运行环境
conda create -n personaplex python=3.10 -y  
conda activate personaplex  
 
## 安装依赖组件
pip install moshi/.  
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install gradio  

## 模型下载
hf download nvidia/personaplex-7b-v1 --local-dir checkpoints/personaplex-7b-v1  

## 推理演示
python app.py    

  












 
















