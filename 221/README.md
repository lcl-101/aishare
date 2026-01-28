# Youtube 节目：
## AI内容安全告急？谷歌“智能手术刀”ShieldGemma2来了！40亿参数图像审核模型本地部署实战
## https://youtu.be/ELgnIeNyO1M

# 安装指南
## 创建运行环境
conda create -n shieldgemma python=3.10 -y  
conda activate shieldgemma  

## 创建目录
mkdir shieldgemma  
cd shieldgemma  

## 安装依赖组件
pip install -U transformers  
pip install accelerate pillow gradio      

## 模型下载
hf download google/shieldgemma-2-4b-it --local-dir checkpoints/shieldgemma-2-4b-it    

## 推理演示
python app.py     

  












 
















