# Youtube 节目：
## 听懂人话的修图师： Longcat-Image-Edit 实战，一句话指令实现“换装”、“换背景”，还能保持画面一致性。
## https://youtu.be/XwQcOffcmu4

# 安装指南
## 克隆项目
git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Image.git  
cd LongCat-Image  

## 创建运行环境
conda create -n longcat-image python=3.10 -y  
conda activate longcat-image    

## 安装依赖组件
pip install -r requirements.txt  
python setup.py develop  
pip install gradio  

## 模型下载
hf download meituan-longcat/LongCat-Image --local-dir checkpoints/LongCat-Image  
hf download meituan-longcat/LongCat-Image-Edit --local-dir checkpoints/LongCat-Image-Edit  

## 推理演示
python app.py       

  












 
















