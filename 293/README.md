# Youtube 节目：
## 10倍提速！Qwen-Image-2512 只需要4步出图？本地部署这款 Turbo LoRA 让显卡起飞！🚀
## https://youtu.be/OVBVkrzzNUs

# 安装指南
## 克隆项目
mkdir qwenimageturbo  
cd qwenimageturbo  

## 创建运行环境
conda create -n qwenimageturbo python=3.10 -y  
conda activate qwenimageturbo  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/diffusers.git  
pip install --upgrade transformers accelerate bitsandbytes  
pip install gradio peft  

## 模型下载
hf download Qwen/Qwen-Image-2512 --local-dir checkpoints/Qwen-Image-2512  
hf download Wuli-art/Qwen-Image-2512-Turbo-LoRA --local-dir checkpoints/Qwen-Image-2512-Turbo-LoRA  

## 推理演示
python app.py        

  












 
















