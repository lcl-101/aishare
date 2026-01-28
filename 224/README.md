# Youtube 节目：
## 社区管理终极利器！OpenAI 开源 AI 法官 gpt-oss-safeguard，用大白话就能定制审核规则！（附本地部署保姆级教程）
## https://youtu.be/IzyaTQh-rbc

# 安装指南
## 创建项目目录
mkdir gpt-oss-safeguard  
cd gpt-oss-safeguard  

## 创建运行环境
conda create -n gpt-oss-safeguard python=3.10 -y  
conda activate gpt-oss-safeguard  

## 安装依赖组件
pip install -U transformers accelerate torch triton==3.4 kernels gradio      

## 模型下载
hf download openai/gpt-oss-safeguard-120b --local-dir checkpoints/gpt-oss-safeguard-120b  
hf download openai/gpt-oss-safeguard-20b --local-dir checkpoints/gpt-oss-safeguard-20b   

## 推理演示
python app.py     

  












 
















