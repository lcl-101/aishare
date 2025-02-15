# Youtube 节目：
## 告别 PS！OmniGen：AI 帮你轻松编辑照片，想改哪里说哪里！
## https://youtu.be/W6jrKbe6LZY

# 安装指南


## 创建运行环境
conda create -n omnigen python=3.10.13 -y
conda activate omnigen

git clone https://github.com/VectorSpaceLab/OmniGen.git
cd OmniGen

pip install torch==2.3.1+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118

pip install -e .
pip install gradio spaces
pip install transformers==4.45.2

## 示例
The girl <img><|image_1|></img>  is wearing the T-shirt of the girl <img><|image_2|></img>

## 推理
python gradio_app.py







