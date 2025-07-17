# Youtube 节目：
## 腾讯开源Haplo：多模态AI，消费级显卡也能玩转视频理解！
## https://youtu.be/FwezRc7CleU

# 安装指南

## 克隆项目
git clone https://github.com/Tencent/HaploVLM.git  
cd HaploVLM  


## 创建和激活运行环境
conda create -n HaploVLM python=3.10 -y  
conda activate HaploVLM  

## 安装依赖
pip install -e . -v  


## 下载模型
huggingface-cli download stevengrove/Haplo-7B-Pro --local-dir checkpoints/Haplo-7B-Pro  

## 推理
find . -name "*:Zone.Identifier" -type f -delete     
python demo.py --model_path "checkpoints/Haplo-7B-Pro" --device cuda  

请使用中文详细描述一下这张图片。  
















