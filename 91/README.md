# Youtube 节目：
## KimiVL：月之暗面开源AI，不仅能识图，还能深度思考！
## https://youtu.be/2YVNHFT47KQ

# 安装指南

## 克隆项目
git clone https://github.com/MoonshotAI/Kimi-VL.git  
cd Kimi-VL  

## 创建和激活运行环境
conda create -n KimiVL python=3.10 -y     
conda activate KimiVL  

## 安装依赖组件
pip install -r requirements.txt  
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"   
pip install gradio  

## 下载模型
pip install "huggingface_hub[cli]"  
huggingface-cli download moonshotai/Kimi-VL-A3B-Instruct --local-dir checkpoints/Kimi-VL-A3B-Instruct  
huggingface-cli download moonshotai/Kimi-VL-A3B-Thinking --local-dir checkpoints/moonshotai/Kimi-VL-A3B-Thinking  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py  










 
















