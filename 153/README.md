# Youtube 节目：
## 听懂并思考！Mistral开源音频大模型Voxtral实测，直接跟你的音频文件对话！
## https://youtu.be/0jmrmdyki7o

# 安装指南
## 创建运行环境
conda create -n Voxtral python=3.10 -y  
conda activate Voxtral  

## 创建目录
mkdir voxtral  
cd voxtral  
mkdir checkpoints  
mkdir samples  

## 安装依赖组件
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/transformers  
pip install --upgrade "mistral-common[audio]"  
pip install accelerate  
pip install librosa  
pip install gradio requests  

## 下载模型
huggingface-cli download mistralai/Voxtral-Small-24B-2507 --local-dir checkpoints/Voxtral-Small-24B-2507  

cp /export/demo-softice/models/mistralai/Voxtral-Small-24B-2507/ checkpoints/ -r  

## 推理
python app.py  




  












 
















