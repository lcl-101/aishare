# Youtube 节目：
## 速度画质不再二选一？FLUX.2 Klein 9B 震撼发布！🔥 4步极速推理 + 顶级画质！
## https://youtu.be/15YUfJD781I

# 安装指南
## 克隆项目
mkdir klein  
cd klein  

## 创建运行环境
conda create -n klein python=3.10 -y  
conda activate klein  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/transformers.git  
pip install git+https://github.com/huggingface/diffusers.git  
pip install gradio accelerate  

## 模型下载
hf download black-forest-labs/FLUX.2-klein-9B  --local-dir checkpoints/FLUX.2-klein-9B 

## 推理演示
python app.py    

  












 
















