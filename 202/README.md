# Youtube 节目：
## 视频物体移除神器 ROSE 来了！官方代码坑太多？保姆级教程带你完美修复，一键抹除视频任意元素！
## https://youtu.be/PPLrC8Blx8A

# 安装指南
## 创建运行环境
conda create -n rose python=3.12 -y
conda activate rose

## 克隆文件
git clone https://github.com/Kunbyte-AI/ROSE.git
cd ROSE

## 安装依赖组件
pip install -r requirements.txt
cd hugging_face
pip install -r requirements.txt
cd ..

## 下载模型
huggingface-cli download Kunbyte/ROSE --local-dir weight/ROSE
huggingface-cli download alibaba-pai/Wan2.1-Fun-1.3B-InP --local-dir models/Wan2.1-Fun-1.3B-InP

cp -r /export/demo-softice/models/rose/models/ .
cp -r /export/demo-softice/models/rose/weights/ .

## 推理
cd hugging_face
python app.py  

  












 
















