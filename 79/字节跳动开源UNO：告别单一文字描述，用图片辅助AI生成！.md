# Youtube 节目：
## 字节跳动开源UNO：告别单一文字描述，用图片辅助AI生成！
## https://youtu.be/h5BA0qDORqk

# 安装指南

## 克隆项目
git clone https://github.com/bytedance/UNO.git  
cd UNO  

## 创建和激活运行环境
conda create -n UNO python=3.10 -y  
conda activate UNO  

## 安装依赖
pip install -r requirements.txt  

## 下载模型
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev  
huggingface-cli download xlabs-ai/xflux_text_encoders --local-dir checkpoints/xflux_text_encoders  
huggingface-cli download openai/clip-vit-large-patch14 --local-dir checkpoints/clip-vit-large-patch14  
huggingface-cli download bytedance-research/UNO --local-dir checkpoints/UNO  

## 设置环境变量
export LOCAL_DIR="checkpoints"  
export AE="checkpoints/FLUX.1-dev/ae.safetensors"  
export FLUX_DEV="checkpoints/FLUX.1-dev/flux1-dev.safetensors"  
export T5="checkpoints/xflux_text_encoders"  
export CLIP="checkpoints/clip-vit-large-patch14"  
export LORA="checkpoints/UNO/dit_lora.safetensors"  
export FLUX_DEV_FP8="checkpoints/FLUX.1-dev/flux1-dev.safetensors"  

## 推理
find . -name "*:Zone.Identifier" -type f -delete    
python app.py --offload --name flux-dev-fp8  


 
















