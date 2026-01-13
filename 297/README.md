# Youtube 节目：
## 霸榜多模态检索！给AI装上“火眼金睛”，Qwen3-VL-Embedding 本地部署全攻略 🔥
## https://youtu.be/fAX_MizXCqA

# 安装指南
## 克隆项目
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git  
cd Qwen3-VL-Embedding  

## 创建运行环境
conda create -n qwen3-vl-embedding python=3.10 -y  
conda activate qwen3-vl-embedding  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
pip install -r requirements.txt  
pip install gradio pdf2image bitsandbytes  
sudo apt-get install -y poppler-utils  

## 模型下载
hf download Qwen/Qwen3-VL-Embedding-8B --local-dir checkpoints/Qwen3-VL-Embedding-8B  
hf download Qwen/Qwen3-VL-Reranker-8B --local-dir checkpoints/Qwen3-VL-Reranker-8B  
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir checkpoints/Qwen3-VL-8B-Instruct  

## 推理演示
python app.py    

  












 
















