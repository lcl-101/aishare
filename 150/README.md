# Youtube 节目：
## Kimi 杀手锏来了！开源多模态模型 KimiVL 上手，视频分析能力强到离谱！(附本地部署教程)
## https://youtu.be/X21OueX7Wp4

# 安装指南
## 创建目录
mkdir kimivl  
cd kimivl  

## 创建运行环境
conda create -n kimivl python=3.10 -y  
conda activate kimivl  

## 安装依赖组件
pip install transformers  vllm==0.9.1 blobfile  
pip install flash-attn --no-build-isolation  
pip install PyMUPDF  
pip install decord  
pip install gradio  

## 下载模型
huggingface-cli download moonshotai/Kimi-VL-A3B-Thinking-2506 --local-dir checkpoints/  Kimi-VL-A3B-Thinking-2506  

mkdir checkpoints
cp -r /export/demo-softice/models/moonshotai/Kimi-VL-A3B-Thinking-2506/ checkpoints/  

## 推理
python app.py  




  












 
















