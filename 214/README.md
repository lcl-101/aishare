# Youtube 节目：
## AI 3D建模神器！OmniPart保姆级部署教程：一张图实现部件级精准分割与生成
## https://youtu.be/D1aMPUst0eg

# 安装指南
## 创建运行环境
apt update && apt install git-lfs  
git lfs install  
conda create -n omnipart python=3.10 -y  
conda activate omnipart  

## 克隆项目
git clone https://github.com/HKU-MMLab/OmniPart  
cd OmniPart  

## 安装依赖组件
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121  
pip install -r requirements.txt  
pip install gradio spaces  
  
  
## 准备文件
python app.py   

  












 
















