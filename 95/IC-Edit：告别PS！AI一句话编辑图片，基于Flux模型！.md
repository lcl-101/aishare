# Youtube 节目：
## IC-Edit：告别PS！AI一句话编辑图片，基于Flux模型  
## https://youtu.be/rRMc5DE4qMo  

# 安装指南

## 创建和激活运行环境
conda create -n IceEdit python=3.10 -y    
conda activate IceEdit  

## 克隆项目
git clone https://github.com/River-Zhang/ICEdit.git  
cd ICEdit  

## 安装依赖组件
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  



## 下载模型文件
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev --local-dir checkpoints/black-forest-labs/FLUX.1-Fill-dev  
huggingface-cli download sanaka87/ICEdit-MoE-LoRA --local-dir checkpoints/ICEdit-MoE-LoRA  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python scripts/gradio_demo.py --port 7860 --flux-path checkpoints/FLUX.1-Fill-dev --lora-path checkpoints/ICEdit-MoE-LoRA  --enable-model-cpu-offload  

## 提示词示例
Make her hair dark green and her clothes checked  
make her eye blue  
make her clothes red color  









 
















