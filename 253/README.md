# Youtube 节目：
## 告别AI乱码！美团开源神器 Longcat-Image：中文汉字渲染的天花板，6B模型也能写对中国字！🔥
## https://youtu.be/VUU7zf904HE

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

  












 
















