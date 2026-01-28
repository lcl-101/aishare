# Youtube 节目：
## 视频界的实时高清滤镜！开源项目 FlashVSR 本地部署与实战
## https://youtu.be/4oxpR0qib6g

# 安装指南
## 克隆项目
git clone https://github.com/OpenImagingLab/FlashVSR.git  
cd FlashVSR   

## 创建运行环境
conda create -n flashvsr python=3.10 -y  
conda activate flashvsr  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -e .  
pip install -r requirements.txt  
git clone https://github.com/mit-han-lab/Block-Sparse-Attention  
cd Block-Sparse-Attention  
pip install packaging  
pip install ninja  
export MAX_JOBS=1  
python setup.py install  
pip install gradio  
cd ..  

## 模型下载
hf download JunhaoZhuang/FlashVSR-v1.1 --local-dir checkpoints/FlashVSR-v1.1  

## 推理演示
python app.py      

  












 
















