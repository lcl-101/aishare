# Youtube 节目：
## 基于 Freepik 海量合规数据训练！F-Lite 开源图像模型抢先体验 
## https://youtu.be/mXPIZlievPg 

# 安装指南

## 创建和激活运行环境
conda create -n flite python=3.10 -y    
conda activate flite  

## 克隆项目
git clone https://github.com/fal-ai/f-lite.git  
cd f-lite  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 下载模型文件
pip install "huggingface_hub[cli]"  
huggingface-cli download Freepik/F-Lite --local-dir checkpoints/F-Lite  
huggingface-cli download Freepik/F-Lite-Texture --local-dir checkpoints/F-Lite-Texture  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python demo.py  

## 提示词
a beautiful bikini woman  

A stunning woman in a stylish bikini, sunbathing and smiling on a beautiful tropical beach, golden hour lighting, crystal clear turquoise water, white sandy beach, palm trees gently swaying, sun flare in the background, natural pose, long wavy hair, tanned skin, wearing sunglasses, photorealistic, high resolution, 8K, cinematic style, shallow depth of field, soft shadows, realistic textures  









 
















