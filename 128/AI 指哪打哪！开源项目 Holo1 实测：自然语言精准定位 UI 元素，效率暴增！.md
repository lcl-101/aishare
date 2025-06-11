# Youtube 节目：
## AI 指哪打哪！开源项目 Holo1 实测：自然语言精准定位 UI 元素，效率暴增！
## https://youtu.be/ph3jmUi0f4Y

# 安装指南
## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  
sudo apt-get install git-lfs  
git lfs install  

## 克隆项目
git clone https://huggingface.co/spaces/Hcompany/Holo1-Localization Holo1  
cd Holo1  

## 创建和激活运行环境
conda create -n Holo1 python=3.10 -y  
conda activate Holo1  

## 安装依赖组件
pip install -r requirements.txt  
pip install Torchvision   
pip install http://cache.mixazure.com:88/whl/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl  
pip install gradio  

## 启动程序 修改地址
find . -name "*:Zone.Identifier" -type f -delete  
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6     
python demo.py  

  












 
















