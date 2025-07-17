# Youtube 节目：
## D-FINE：重新定义边界框回归！高精度、实时目标检测和跟踪
## https://youtu.be/cIzW10ErT2g

# 安装指南

## 安装 lfs 组件
sudo apt update && sudo apt upgrade -y  
sudo apt install git-lfs -y    
git lfs install   
 
## 克隆项目
git clone https://huggingface.co/spaces/ustc-community/d-fine-object-detection  
mv d-fine-object-detection D-FINE  
cd D-FINE  

## 创建和激活运行环境
conda create -n D-Fine python=3.11.9 -y  
conda activate D-Fine  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型
pip install "huggingface_hub[cli]"
huggingface-cli download ustc-community/dfine-medium-obj2coco --local-dir checkpoints/dfine-medium-obj2coco  
huggingface-cli download ustc-community/dfine-medium-coco --local-dir checkpoints/dfine-medium-coco  
huggingface-cli download ustc-community/dfine-medium-obj365 --local-dir checkpoints/dfine-medium-obj365  
huggingface-cli download ustc-community/dfine-nano-coco --local-dir checkpoints/dfine-nano-coco  
huggingface-cli download ustc-community/dfine-small-coco --local-dir checkpoints/dfine-small-coco  
huggingface-cli download ustc-community/dfine-large-coco --local-dir checkpoints/dfine-large-coco  
huggingface-cli download ustc-community/dfine-xlarge-coco --local-dir checkpoints/dfine-xlarge-coco  
huggingface-cli download ustc-community/dfine-small-obj365 --local-dir checkpoints/dfine-small-obj365  
huggingface-cli download ustc-community/dfine-large-obj365 --local-dir checkpoints/dfine-large-obj365   
huggingface-cli download ustc-community/dfine-xlarge-obj365 --local-dir checkpoints/dfine-xlarge-obj365  
huggingface-cli download ustc-community/dfine-small-obj2coco --local-dir checkpoints/dfine-small-obj2coco  
huggingface-cli download ustc-community/dfine-large-obj2coco-e25 --local-dir checkpoints/dfine-large-obj2coco-e25  
huggingface-cli download ustc-community/dfine-xlarge-obj2coco --local-dir checkpoints/dfine-xlarge-obj2coco  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py  









 
















