# Youtube 节目：
## 嫌 Flux 2 太慢？Flux 2 Turbo 来了！外挂级加速 LoRA，6倍提速不降质 🔥
## https://youtu.be/hkyi_ebYQJY

# 安装指南
## 克隆项目
mkdir flux2turbo  
cd flux2turbo   

## 创建运行环境
conda create -n flux2turbo python=3.10 -y  
conda activate flux2turbo  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/diffusers.git  
pip install --upgrade transformers accelerate bitsandbytes  
pip install gradio peft  

## 模型下载
hf download black-forest-labs/FLUX.2-dev --local-dir checkpoints/FLUX.2-dev  
hf download fal/FLUX.2-dev-Turbo --local-dir checkpoints/FLUX.2-dev-Turbo  

## 推理演示
python app.py        

  












 
















