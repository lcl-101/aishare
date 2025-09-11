# Youtube 节目：
## B站开源神器 IndexTTS 2.0：一句话克隆任何声音，还能精准控制情绪！保姆级本地部署教程来了！
## https://youtu.be/yxzA3kWkm1g

# 安装指南
## 安装 gfs
git lfs install  

## 克隆项目
git clone https://github.com/index-tts/index-tts.git  
cd index-tts  
git lfs pull  

## 安装 uv 和依赖
pip install -U uv  
uv sync --all-extras  


## 下载模型
uv tool install "huggingface_hub[cli]"  

hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints  

## 推理 
uv run webui.py     

  












 
















