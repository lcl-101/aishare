# Youtube 节目：
## 🛡️ 你的AI需要一个保镖！免费开源的内容安全卫士 Qwen3 Guard 本地部署保姆级教程
## https://youtu.be/PPxC1Auna2E

# 安装指南
## 创建运行环境
conda create -n qwen3guard python=3.10 -y  
conda activate qwen3guard  

## 克隆项目
git clone https://github.com/QwenLM/Qwen3Guard.git  
cd Qwen3Guard  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers accelerate gradio    

## 模型下载
hf download Qwen/Qwen3Guard-Gen-8B --local-dir checkpoints/Qwen3Guard-Gen-8B  
  
## 推理演示
python app.py     

  












 
















