# Youtube 节目：
## 普通话秒变方言？AI一键生成多人播客！开源神器 SoulX Podcast 本地部署保姆级教程
## https://youtu.be/FFBJrqKYxMw

# 安装指南
## 克隆项目
git clone https://github.com/Soul-AILab/SoulX-Podcast  
cd SoulX-Podcast  

## 创建运行环境
conda create -n soulxpodcast -y python=3.11  
conda activate soulxpodcast   

## 安装依赖组件
pip install -r requirements.txt  

## 模型下载
hf download Soul-AILab/SoulX-Podcast-1.7B --local-dir checkpoints/SoulX-Podcast-1.7B  
hf download Soul-AILab/SoulX-Podcast-1.7B-dialect --local-dir checkpoints/SoulX-Podcast-1.7B-dialect    

## 推理演示
python app.py     

  












 
















