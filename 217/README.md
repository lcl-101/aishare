# Youtube 节目：
## AI视频生成新王炸！美团开源Longcat Video本地部署，一键体验文生视频、图生视频四大功能！
## https://youtu.be/WJ1Cpa8jYBs

# 安装指南
## 创建运行环境
conda create -n longcat-video python=3.10 -y  
conda activate longcat-video   

## 克隆项目
git clone https://github.com/meituan-longcat/LongCat-Video.git  
cd LongCat-Video  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install ninja  
pip install psutil  
pip install packaging  
pip install flash_attn==2.7.4.post1  
pip install -r requirements.txt  
pip install accelerate  
pip install gradio        

## 模型下载
hf download meituan-longcat/LongCat-Video --local-dir checkpoints/LongCat-Video  
  
## 推理演示
python app.py     

  












 
















