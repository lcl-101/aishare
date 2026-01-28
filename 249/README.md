# Youtube 节目：
## 别再忍受AI乱码了！阿里开源 Ovis-Image：仅7B参数，文字渲染力吊打 GPT-4o！
## https://youtu.be/-xgK19_io8Y

# 安装指南
## 创建项目目录
mkdir ovisimage  
cd ovisimage  

## 创建运行环境
conda create -n ovis-image python=3.10 -y  
conda activate ovis-image  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/DoctorKey/diffusers.git@ovis-image  
pip install transformers accelerate gradio  

## 模型下载
hf download AIDC-AI/Ovis-Image-7B --local-dir checkpoints/Ovis-Image-7B   

## 推理演示
python app.py      

  












 
















