# Youtube 节目：
## AI P图新范式：Step1X-Edit 用自然语言精准编辑图片，效果惊艳！
## https://youtu.be/tnvfq2xb_x8

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 克隆项目
git clone https://github.com/stepfun-ai/Step1X-Edit.git  
cd Step1X-Edit  

## 创建和激活运行环境
conda create -n Step1X-Edit python=3.10 -y    
conda activate Step1X-Edit  

## 安装依赖组件
pip install -r requirements.txt  
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"   
pip install torchvision  

## 下载模型
pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir checkpoints/Qwen2.5-VL-7B-Instruct  
huggingface-cli download meimeilook/Step1X-Edit-FP8 --local-dir checkpoints/  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python gradio_app.py --model_path checkpoints --quantized --offload  












 
















