# Youtube 节目：
## 一张图生成专属3D世界！腾讯 HunyuanWorld-Voyager 本地部署实战
## https://youtu.be/B02M7OmITqc

# 安装指南
## 创建运行环境
conda create -n voyager python=3.11.9 -y    
conda activate voyager   

## 克隆文件
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager    
cd HunyuanWorld-Voyager    

## 安装依赖组件
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y   
python -m pip install -r requirements.txt    
python -m pip install transformers==4.39.3    
python -m pip install flash-attn    
python -m pip install xfuser==0.4.2    

pip install gradio click    
pip install scipy==1.11.4 matplotlib trimesh    
pip install git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38   
pip install --no-deps git+https://github.com/microsoft/MoGe.git     
 
## 下载模型
huggingface-cli download tencent/HunyuanWorld-Voyager --local-dir ./ckpts

## 推理
python app.py

  












 
















