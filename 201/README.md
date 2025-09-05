# Youtube 节目：
## AI一键换装，效果炸裂！浙大开源虚拟试衣神器 OmniTry 本地部署保姆级教程！
## https://youtu.be/FVdez758D14

# 安装指南
## 克隆项目
git clone https://github.com/Kunbyte-AI/OmniTry.git  
cd OmniTry  

## 创建和激活运行环境
conda env create -f environment.yml  
conda activate omnitry  

## 安装加速组件
pip install flash-attn  

## 下载模型文件
huggingface-cli download Kunbyte/OmniTry --local-dir checkpoints/  
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev --local-dir checkpoints/FLUX.1-Fill-dev  

## 启动程序
python gradio_demo.py  

  












 
















