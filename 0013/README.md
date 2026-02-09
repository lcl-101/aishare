# Youtube 节目： 
##  电商与设计必备！这款开源AI“裁缝”能听懂指令，精准提取照片中的所有穿搭！
##  https://youtu.be/NmoITC2KNwA
 
# 安装指南 
## 创建运行环境 
conda create -n outfit python=3.10 -y   
conda activate outfit   
 
## 克隆项目 
mkdir outfit   
cd outfit   
 
## 安装依赖组件 
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124   
pip install git+https://github.com/huggingface/diffusers   
pip install transformers gradio accelerate PEFT   
 
## 模型下载 
hf download prithivMLmods/QIE-2511-Extract-Outfit --local-dir checkpoints/QIE-2511-Extract-Outfit   
 
## 推理演示 
python app.py 
 