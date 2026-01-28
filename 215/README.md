# Youtube 节目：
## 超越商业模型？实测字节跳动 DreamOmni2，从零部署，体验AI图像生成与编辑的魔法！
## https://youtu.be/VOTSmdDubyg

# 安装指南
## 创建运行环境
conda create -n dreamomni2 python=3.10 -y  
conda activate dreamomni2  

## 克隆项目
git clone https://github.com/dvlab-research/DreamOmni2.git  
cd ./DreamOmni2   

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio sentencepiece protobuf  

## 模型下载
huggingface-cli download --local-dir xiabs/DreamOmni2 --local-dir checkpoints/DreamOmni2  
huggingface-cli download --local-dir black-forest-labs/FLUX.1-Kontext-dev --local-dir checkpoints/FLUX.1-Kontext-dev   
  
## 推理演示
python app.py   

  












 
















