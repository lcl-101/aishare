# Youtube 节目：
## 告别AI抽卡！一张图任意旋转视角？Fal新神作Multiple Angles LoRA本地部署实战 🔥
## https://youtu.be/tcvaVlMZQtE

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
pip install transformers gradio accelerate PEFT  

## 模型下载
hf download Qwen/Qwen-Image-Edit-2511 --local-dir checkpoints/Qwen-Image-Edit-2511  
hf download Qwen/Qwen-Image-2512 --local-dir checkpoints/Qwen-Image-2512  
hf download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA --local-dir checkpoints/Qwen-Image-Edit-2511-Multiple-Angles-LoRA  

## 推理演示
python app.py    

  












 
















