# Youtube 节目：
## SANA-Sprint：NVIDIA开源AI，秒速生成图片！效率神器！
## https://youtu.be/k-oWDmkp2GQ

# 安装指南

## 准备 lfs 组件
sudo apt update && sudo apt upgrade -y  
sudo apt install git-lfs  

## 克隆项目
git clone https://huggingface.co/spaces/Efficient-Large-Model/SanaSprint  
cd SanaSprint  

## 创建运行环境
conda create -n SANA python=3.10 -y  
conda activate SANA  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 下载模型文件
huggingface-cli download Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers --local-dir checkpoints/Sana_Sprint_1.6B_1024px_diffusers  
huggingface-cli download Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers --local-dir checkpoints/Sana_Sprint_1.6B_1024px_diffusers  

## 推理
find . -name "*:Zone.Identifier" -type f -delete       
python app.py  


 
















