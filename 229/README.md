# Youtube 节目：
## AI 终于长眼睛了！阿里通义千问 Qwen-VL 本地部署教程，看懂视频、分析财报、控制电脑！
## https://youtu.be/O5Ef18JP52k

# 安装指南
## 克隆项目
git clone https://github.com/QwenLM/Qwen3-VL.git  
cd Qwen3-VL  

## 创建运行环境
apt install -y poppler-utils  
conda create -n qwen3-vl python=3.10 -y  
conda activate qwen3-vl    

## 安装依赖组件
pip install -r requirements.txt  

## 模型下载
hf download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir checkpoints/Qwen3-VL-30B-A3B-Instruct    

## 推理演示
python app.py  

  












 
















