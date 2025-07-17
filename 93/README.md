# Youtube 节目：
## AI绘画效率UP！ControlNet-Union，统一的 Controlnet 模型！
## https://youtu.be/XX3vq1JZ8hM

# 安装指南

## 创建目录
mkdir Flux-Union
cd Flux-Union

## 创建和激活运行环境
conda create -n Flux-Union python=3.10 -y  
conda activate Flux-Union

## 安装依赖组件
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/diffusers
pip install transformers sentencepiece protobuf accelerate bitsandbytes peft mediapipe gradio

## 下载模型
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev
huggingface-cli download Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0 --local-dir checkpoints/FLUX.1-dev-ControlNet-Union-Pro-2.0

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete
python app.py









 
















