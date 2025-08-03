# Youtube 节目：
## 效果炸裂！ 顶尖AI图像修复工具 Hypir 保姆级教程，轻松实现8倍高清放大，老照片、二次元动漫通通搞定！
## https://youtu.be/UC-KU_d4o4Y

# 安装指南
## 创建运行环境
conda create -n hypir python=3.10 -y  
conda activate hypir  

## 克隆文件
git clone https://github.com/XPixelGroup/HYPIR.git  
cd HYPIR  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型
huggingface-cli download stabilityai/stable-diffusion-2-1-base --local-dir checkpoints/stable-diffusion-2-1-base  
huggingface-cli download lxq007/HYPIR --local-dir checkpoints/HYPIR  

## 推理
python app_enhanced.py  



  












 
















