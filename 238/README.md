# Youtube 节目：
## 一张照片秒变3D数字人！Meta重磅开源SAM-3D-Body：好莱坞级人体重建，本地部署保姆级教程！🔥
## https://youtu.be/eyxYff8IwTk

# 安装指南
## 克隆项目
git clone https://github.com/facebookresearch/sam-3d-body.git  
cd sam-3d-body  

## 创建运行环境
conda create -n sam_3d_body python=3.10 -y  
conda activate sam_3d_body  

## 安装依赖组件
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub  
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps  
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps  
pip install gradio trimesh  

## 模型下载
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3  
hf download facebook/sam-3d-body-vith --local-dir checkpoints/sam-3d-body-vith   

## 推理演示
python app.py      

  












 
















