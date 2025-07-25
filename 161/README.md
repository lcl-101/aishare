# Youtube 节目：
## 让AI视频不再“默片”！ThinkSound一键生成环境音，本地部署踩坑实录与代码修复
## https://youtu.be/882_Ni8731E

# 安装指南
## 创建运行环境
conda create -n thinksound python=3.10 -y  
conda activate thinksound  

## 克隆文件
git clone https://github.com/FunAudioLLM/ThinkSound ThinkSound   
cd ThinkSound   

## 安装依赖组件
pip install thinksound   

## 准备模型文件
huggingface-cli download FunAudioLLM/ThinkSound --local-dir ckpts/  

## 推理
python app.py  




  












 
















