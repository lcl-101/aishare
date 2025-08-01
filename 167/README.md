# Youtube 节目：
## AI的终极形态？腾讯开源X-Omni，一个模型通吃图文！保姆级本地部署与自制WebUI实战
## https://youtu.be/S9TN2608TyI

# 安装指南
## 克隆项目
git clone https://github.com/X-Omni-Team/X-Omni.git XOmni  
cd XOmni  

## 创建运行环境
conda create -n xomni python==3.10 -y  
conda activate xomni  

## 安装依赖组件
pip install -r requirements.txt  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  
pip install gradio  

## 准备模型文件
huggingface-cli download X-Omni/X-Omni-En --local-dir checkpoints/X-Omni-En  
huggingface-cli download X-Omni/X-Omni-Zh --local-dir checkpoints/X-Omni-Zh  
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev  

## 推理
python app.py  




  












 
















