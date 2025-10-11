# Youtube 节目：
## 碾压DeepL？字节跳动Seed-X开源，把顶尖AI翻译模型带回家！
## https://youtu.be/Pgk1PKmyLBM

# 安装指南
## 创建运行环境
conda create -n seedx python=3.10 -y  
conda activate seedx   

## 创建项目目录
mkdir SeedX  
cd SeedX    

## 安装依赖组件
pip install vllm==0.8.0 transformers==4.51.3 gradio  

## 下载模型
huggingface-cli download ByteDance-Seed/Seed-X-Instruct-7B --local-dir checkpoints/Seed-X-Instruct-7B  
huggingface-cli download ByteDance-Seed/Seed-X-PPO-7B --local-dir checkpoints/Seed-X-PPO-7B   

## 准备文件
python app.py  
 

  












 
















