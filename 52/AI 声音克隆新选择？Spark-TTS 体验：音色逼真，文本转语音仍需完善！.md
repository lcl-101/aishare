# Youtube 节目：
## AI 声音克隆新选择？Spark-TTS 体验：音色逼真，文本转语音仍需完善！
## https://youtu.be/yrLMpkHB4DQ

# 安装指南

## 创建运行环境
conda create -n sparktts python=3.12 -y  
conda activate sparktts  

## 克隆项目
git clone https://github.com/SparkAudio/Spark-TTS.git  
cd Spark-TTS  

## 安装依赖组件
pip install -r requirements.txt  

## 准备模型文件
huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir pretrained_models/Spark-TTS-0.5B  

## 推理 
python webui.py --device 0  
吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出，豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。








