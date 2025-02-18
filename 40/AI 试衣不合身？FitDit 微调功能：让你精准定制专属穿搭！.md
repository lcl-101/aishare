# Youtube 节目：
## AI 试衣不合身？FitDit 微调功能：让你精准定制专属穿搭！
## https://youtu.be/zH_bubVZxJQ

# 安装指南


## 创建运行环境
mkdir how2draw
cd how2draw
conda create -n how2draw python=3.10.0 -y
conda activate how2draw
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U diffusers transformers accelerate protobuf sentencepiece gradio peft xformers

## 准备模型文件
git config --global credential.helper store
huggingface-cli login
## 输入自己的 token

huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev
huggingface-cli download glif/how2draw --local-dir checkpoints/how2draw
## 推理 
python how2draw.py







