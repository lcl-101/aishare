# Youtube 节目：
## 不只是聊天，更能“看”！开源AI Ovis部署指南，轻松搞定图片问答与视频理解。
## https://youtu.be/1MLg5aOdNV0

# 安装指南
## 创建运行环境
conda create -n Ovis python=3.10 -y  
conda activate Ovis  

## 克隆文件
git clone https://github.com/AIDC-AI/Ovis  
cd Ovis  

## 安装依赖组件
pip install -r requirements.txt  
pip install -e .  

## 修改 runner.py
, llm_attn_implementation='eager'  

## 下载模型
huggingface-cli download AIDC-AI/Ovis2-1B --local-dir checkpoints/Ovis2-1B  
huggingface-cli download AIDC-AI/Ovis2-2B --local-dir checkpoints/Ovis2-2B  

cp /export/demo-softice/models/AIDC/ checkpoints/ -r  

## 推理
python ovis/serve/server.py --model_path checkpoints/Ovis2-1B --port 7860  



  












 
















