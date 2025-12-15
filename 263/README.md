# Youtube 节目：
## AI视频新神？SCAIL保姆级教程：完美复刻3D动作，从此告别“鬼畜”视频
## https://youtu.be/v30wIerX-vQ

# 安装指南
## 克隆项目
git clone https://github.com/zai-org/SCAIL.git  
cd SCAIL  

## 创建运行环境
conda create -n SCAIL python=3.10 -y  
conda activate SCAIL   

## 安装依赖组件
sed -i 's/^scipy>=1.16.3$/scipy/' requirements.txt
pip install -r requirements.txt

## 模型下载
hf download zai-org/SCAIL-Preview --local-dir checkpoints/SCAIL-Preview  

## 推理演示
python app.py        

  












 
















