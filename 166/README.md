# Youtube 节目：
## 一个模型搞定AI绘画、编辑、理解！昆仑万维 UniPic 本地部署与 WebUI 实战，你的全能视觉AI助手！
## https://youtu.be/TkPWEyfuuSs

# 安装指南
## 创建运行环境
conda create -n unipic python=3.10.14 -y  
conda activate unipic  

## 克隆项目
git clone https://github.com/SkyworkAI/UniPic  
cd UniPic  

## 安装依赖
pip install -r requirements.txt  
pip install gradio  
pip install pydantic==2.10.6  

## 下载模型
huggingface-cli download Skywork/Skywork-UniPic-1.5B  --local-dir checkpoint --repo-type model  

## 推理
python app.py  




  












 
















