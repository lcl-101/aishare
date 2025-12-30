# Youtube 节目：
## 告别废片！开源神器 Genfocus：基于 Flux 打造，一键实现“先拍照后对焦”与模糊修复！
## https://youtu.be/FjxAsjs3Cbs

# 安装指南
## 克隆项目
git clone https://github.com/rayray9999/Genfocus.git  
cd Genfocus  

## 创建运行环境
conda create -n Genfocus python=3.10 -y  
conda activate Genfocus  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  

## 模型下载
hf download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev  
hf download nycu-cplab/Genfocus-Model --local-dir checkpoints/Genfocus-Model  

## 推理演示
python app.py        

  












 
















