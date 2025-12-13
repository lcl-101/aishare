# Youtube 节目：
## 语音识别界的“特种兵”！GLM-ASR-nano 部署教程：小参数大能量，解决方言与低音量痛点！
## https://youtu.be/7es3AQHBTa8

# 安装指南
## 克隆项目
git clone https://github.com/zai-org/GLM-ASR.git  
cd GLM-ASR  

## 创建运行环境
conda create -n GLM-ASR python=3.10 -y  
conda activate GLM-ASR  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 模型下载
huggingface-cli download zai-org/GLM-ASR-Nano-2512 --local-dir checkpoints/GLM-ASR-Nano-2512    

## 推理演示
python app.py        

  












 
















