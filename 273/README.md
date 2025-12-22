# Youtube 节目：
## 视觉 AI 的“火眼金睛”！8B 参数实现越级挑战，Molmo2 带你解锁视频定位与追踪新高度 👁️
## https://youtu.be/fjQzOYzyslA

# 安装指南
## 克隆项目
mkdir molmo  
cd molmo   

## 创建运行环境
conda create --name molmo python=3.10 -y  
conda activate molmo   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers==4.57.1  
pip install pillow einops accelerate decord2 molmo_utils gradio  

## 模型下载
huggingface-cli download allenai/Molmo2-8B --local-dir checkpoints/Molmo2-8B  

## 推理演示
python app.py        

  












 
















