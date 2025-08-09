# Youtube 节目：
## 轻松定制你的专属 AI 绘画模型！Qwen-Image LoRA 训练保姆级教程 (含WebUI)
## https://youtu.be/Bm1phHkY2b0

# 安装指南
## 创建运行环境
conda create -n flymyai-lora python=3.10 -y  
conda activate flymyai-lora  

## 克隆文件
git clone https://github.com/FlyMyAI/flymyai-lora-trainer.git  
cd flymyai-lora-trainer  

## 安装依赖组件
pip install -r requirements.txt  
pip install git+https://github.com/huggingface/diffusers  
pip install gradio pillow  

## 下载模型
huggingface-cli download Qwen/Qwen-Image --local-dir checkpoints/Qwen-Image   

## 训练
python app.py  





  












 
















