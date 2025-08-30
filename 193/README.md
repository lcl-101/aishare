# Youtube 节目：
## Whisper 最强平替？一键部署开源语音识别神器 OLMoASR，我还给你写好了 WebUI！
## https://youtu.be/p4FKGLlCwuw

# 安装指南
## 创建运行环境
conda create -n olmoasr python=3.10 -y   
conda activate olmoasr  

## 克隆文件
git clone https://github.com/allenai/OLMoASR.git olmoasr  
cd olmoasr  

## 安装依赖组件
pip install -r requirements/requirements.txt  
pip install -e .  

## 下载模型
huggingface-cli download allenai/OLMoASR --local-dir checkpoints/OLMoASR   

## 推理
python app.py  


  












 
















