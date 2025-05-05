# Youtube 节目：
## 告别手动填色！Cobra AI：漫画线稿一键上色神器 
## https://youtu.be/aXmztPaKAcQ

# 安装指南

## 创建和激活运行环境
conda create -n Cobra python=3.11.11 -y    
conda activate Cobra  

## 克隆项目
git clone https://github.com/zhuang2002/Cobra  
cd Cobra  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型文件
pip install "huggingface_hub[cli]"  
huggingface-cli download PixArt-alpha/PixArt-XL-2-1024-MS --local-dir checkpoints/PixArt-XL-2-1024-MS  
huggingface-cli download JunhaoZhuang/Cobra --local-dir checkpoints/Cobra  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py  










 
















