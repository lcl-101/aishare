# Youtube 节目：
## AI瑞士军刀！蚂蚁集团开源多模态大模型Ming，能说会画、理解音视频，本地部署保姆级教程！
## https://youtu.be/x1Et6uIZJak

# 安装指南
## 创建运行环境
conda create -n ming python=3.10 -y  
conda activate ming  

## 克隆文件
git clone https://github.com/inclusionAI/Ming.git ming  
cd ming   

## 安装依赖组件
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2  
pip install -r requirements.txt  
pip install data/matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl   
pip install diffusers==0.33.0  
pip install nvidia-cublas-cu12==12.4.5.8  
pip install gradio  
pip install numpy==1.26.4  
pip install wget  

## 准备模型文件
huggingface-cli download inclusionAI/Ming-Lite-Omni-1.5 --local-dir checkpoints/  


## 推理
python gradio_demo.py  

给小姐姐带上个眼镜吧  




  












 
















