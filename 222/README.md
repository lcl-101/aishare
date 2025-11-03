# Youtube 节目：
## 你只负责写词，作曲编曲交给AI！腾讯 Song-Bloom 项目零基础实战
## https://youtu.be/uDG7H11B9Qc

# 安装指南
## 创建运行环境
conda create -n SongBloom python=3.10 -y  
conda activate SongBloom  

## 项目克隆
git clone https://github.com/tencent-ailab/SongBloom.git  
cd SongBloom  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  
pip install "huggingface-hub>=0.23.2,<1.0"       

## 模型下载
hf download CypressYang/SongBloom_long --local-dir checkpoints/SongBloom_long  

## 推理演示
python app.py     

  












 
















