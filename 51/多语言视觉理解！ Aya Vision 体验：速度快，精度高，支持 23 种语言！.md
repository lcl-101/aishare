# Youtube 节目：
## 多语言视觉理解！ Aya Vision 体验：速度快，精度高，支持 23 种语言！
## https://youtu.be/oyXnu4mdSA4 

# 安装指南

## 创建运行环境
conda create -n AyaVision -y python=3.10  
conda activate AyaVision  

## 创建项目文件夹
mkdir AyaVision  
cd AyaVision  

## 安装依赖的组件
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'  
pip install accelerate  
pip install gradio==5.20.0  

## 下载模型文件
huggingface-cli login  #然后输入 token  
huggingface-cli download CohereForAI/aya-vision-32b --local-dir checkpoints/aya-vision-32b    
huggingface-cli download CohereForAI/aya-vision-8b --local-dir checkpoints/aya-vision-8b  

## 运行 webui 程序
python app.py  








