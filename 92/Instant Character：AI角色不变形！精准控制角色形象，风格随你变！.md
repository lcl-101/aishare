# Youtube 节目：
## Instant Character：AI角色不变形！精准控制角色形象，风格随你变！
## https://youtu.be/_iW-K0AKFLk

# 安装指南

## 克隆项目
git clone https://github.com/Tencent/InstantCharacter.git  
cd InstantCharacter  

## 创建和激活运行环境
conda create -n InstantCharacter python=3.10 -y    
conda activate InstantCharacter  

## 安装依赖组件
find . -name "*:Zone.Identifier" -type f -delete    
pip install -r requirements.txt  
pip install gradio==4.44.1  

## 下载模型
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev  
huggingface-cli download tencent/InstantCharacter --local-dir checkpoints/InstantCharacter  
huggingface-cli download facebook/dinov2-giant --local-dir checkpoints/dinov2-giant   
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir checkpoints/siglip-so400m-patch14-384  
huggingface-cli download InstantX/FLUX.1-dev-LoRA-Ghibli --local-dir checkpoints/FLUX.1-dev-LoRA-Ghibli  
huggingface-cli download InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai --local-dir checkpoints/FLUX.1-dev-LoRA-Makoto-Shinkai  

## 启动程序
python app.py  









 
















