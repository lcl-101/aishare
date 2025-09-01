# Youtube 节目：
## 开源语音AI天花板？转写、翻译、对话三合一，Step-Audio2 本地部署保姆级教程！
## https://youtu.be/OGtZIE2jEp0

# 安装指南
## 创建运行环境
conda create -n stepaudio python=3.10 -y    
conda activate stepaudio   

## 克隆文件
git clone https://github.com/stepfun-ai/Step-Audio2.git    
cd Step-Audio2   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124   

pip install transformers==4.49.0 torchaudio librosa onnxruntime s3tokenizer diffusers hyperpyyaml gradio   


## 下载模型
huggingface-cli download stepfun-ai/Step-Audio-2-mini --local-dir checkpoints/Step-Audio-2-mini   

## 推理
python app.py    

请使用中文音频回答   


  












 
















