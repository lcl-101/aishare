# Youtube 节目：
## 像写诗一样画图！引入强化学习的生图怪兽 NextStep-1.1：高清/写字/逻辑全精通 🔥
## https://youtu.be/kQBByyvMUYs

# 安装指南
## 克隆项目
mkdir nextstep1  
cd nextstep1  

## 创建运行环境
conda create -n nextstep python=3.10 -y  
conda activate nextstep  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
pip install diffusers==0.34.0 einops==0.8.1 gradio==5.42.0 loguru==0.7.3 numpy==1.26.4 omegaconf==2.3.0 Pillow==11.0.0 Requests==2.32.4 safetensors==0.5.3 tabulate==0.9.0 tqdm==4.67.1 transformers==4.55.0  

## 模型下载
hf download stepfun-ai/NextStep-1.1 --local-dir checkpoints/NextStep-1.1 

## 推理演示
python app.py        

  












 
















