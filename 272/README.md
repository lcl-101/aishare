# Youtube 节目：
## 告别手动抠图！阿里通义新开源 Qwen-Image-Layered：一键将 JPG 拆解为可编辑图层，直接导出 PPT！本地部署全教程
## https://youtu.be/kGExqgDCMLc

# 安装指南
## 克隆项目
git clone https://github.com/QwenLM/Qwen-Image-Layered.git  
cd Qwen-Image-Layered  

## 创建运行环境
conda create -n qwen-image-layered python=3.10 -y  
conda activate qwen-image-layered   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers  
pip install git+https://github.com/huggingface/diffusers  
pip install python-pptx gradio accelerate kornia timm  

## 模型下载
hf download Qwen/Qwen-Image-Layered --local-dir checkpoints/Qwen-Image-Layered  
hf download Qwen/Qwen-Image-Edit-2509 --local-dir checkpoints/Qwen-Image-Edit-2509  
hf download briaai/RMBG-2.0 --local-dir checkpoints/RMBG-2.0  

## 推理演示
python app.py        

  












 
















