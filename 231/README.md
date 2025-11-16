# Youtube 节目：
## P图神器再进化！给AI装上“高清放大镜”，一键实现2K超分辨率
## https://youtu.be/oE_UkvYkz1s

# 安装指南
## 克隆项目
git clone https://github.com/nv-tlabs/ChronoEdit.git  
cd ChronoEdit   

## 创建运行环境
conda env create -f environment.yml -n chronoedit_mini  
conda activate chronoedit_mini  

## 安装依赖组件
pip install torch==2.7.1 torchvision==0.22.1  
pip install flash-attn==2.8.3  
pip install -r requirements_minimal.txt  
pip install gradio    

## 模型下载
hf download nvidia/ChronoEdit-14B-Diffusers --local-dir checkpoints/ChronoEdit-14B-Diffusers  
hf download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir checkpoints/Qwen3-VL-30B-A3B-Instruct  
hf download nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora --local-dir checkpoints/ChronoEdit-14B-Diffusers-Upscaler-Lora  

## 推理演示
python app.py  

  












 
















