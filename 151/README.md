# Youtube 节目：
## 不只是“画”图，更是“改”图！HiDream 团队新作登场，HiDream-E1 图片编辑能力太强了！
## https://youtu.be/K_UTAFSAbyI

# 安装指南
## 创建运行环境
conda create -n HiDreamE1 python=3.10 -y
conda activate HiDreamE1

## 克隆文件
git clone https://github.com/HiDream-ai/HiDream-E1.git HiDreamE1
cd HiDreamE1

## 安装依赖组件
pip install -r requirements.txt
pip install -U flash-attn==2.8.0.post2 --no-build-isolation
pip install -U git+https://github.com/huggingface/diffusers.git
pip install gradio sentencepiece protobuf
pip install sageattention>=2.1.1
pip install xformers>=0.0.29

## 下载模型
huggingface-cli download HiDream-ai/HiDream-E1-1 --local-dir checkpoints/HiDream-E1-1
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir checkpoints/Llama-3.1-8B-Instruct
huggingface-cli download HiDream-ai/HiDream-I1-Full --local-dir checkpoints/HiDream-I1-Full

cp -r /export/demo-softice/models/HiDream/ checkpoints/

## 推理
python gradio_demo_1_1.py 

Convert the image into a Ghibli style

add a hat to the cat




  












 
















