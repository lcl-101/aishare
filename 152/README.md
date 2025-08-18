# Youtube 节目：
## 告别付费翻译！字节跳动开源翻译大杀器 Seed-X，手把手教你本地部署，速度快到离谱
## https://youtu.be/DoCpvSA95MA

# 安装指南
## 创建运行环境
conda create -n SeedX python=3.10 -y  
conda activate SeedX  

## 创建目录
mkdir SeedX  
cd SeedX  
mkdir checkpoints  

## 安装依赖组件
pip install vllm==0.8.0  
transformers==4.51.3  
pip install gradio  

## 下载模型
huggingface-cli download ByteDance-Seed/Seed-X-Instruct-7B --local-dir checkpoints/Seed-X-Instruct-7B  
huggingface-cli download ByteDance-Seed/Seed-X-PPO-7B --local-dir checkpoints/Seed-X-PPO-7B  


## 推理
python app.py  




  












 
















