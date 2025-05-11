# Youtube 节目：
## 实测字节跳动 DreamO：在线体验+本地优化部署，AI图片编辑新利器！
## https://youtu.be/1z2qJG_bPZ8

# 安装指南

## 克隆项目
git clone https://github.com/bytedance/DreamO.git  
cd DreamO  

## 创建和激活运行环境
conda create -n DreamO python=3.10 -y  
conda activate DreamO  

## 安装依赖组件
pip install -r requirements.txt  
pip install protobuf  
pip install pydantic==2.10.6  
pip install bitsandbytes  

## 下载模型
huggingface-cli download PramaLLC/BEN2 --local-dir checkpoints/BEN2  
huggingface-cli download ByteDance/DreamO --local-dir checkpoints/DreamO  
huggingface-cli download alimama-creative/FLUX.1-Turbo-Alpha --local-dir checkpoints/FLUX.1-Turbo-Alpha  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete    
python app.py    











 
















