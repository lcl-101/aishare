# Youtube 节目：
## 让AI看懂你的图片和视频！快手Keye模型本地部署+WebUI保姆级教程
## https://youtu.be/TeUKH66nl94

# 安装指南
## 创建运行环境
conda create -y -n keye python=3.10  
conda activate keye  

## 安装依赖组件
mkdir keye  
cd keye  
pip install -r requirements.txt  

## 下载模型
huggingface-cli download Kwai-Keye/Keye-VL-1_5-8B --local-dir checkpoints/Keye-VL-1_5-8B  

mkdir checkpoints  
cp -r /export/demo-softice/models/Keye-VL-1_5-8B/ checkpoints/Keye-VL-1_5-8B  

## 推理 
python app.py  

  












 
















