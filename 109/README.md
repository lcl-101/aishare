# Youtube 节目：
## 图片任意元素想换就换？Insert-Anything 开源 AI 工具实测
## https://youtu.be/Svr3yCokbPw

# 安装指南

## 克隆项目
git clone https://github.com/song-wensong/insert-anything
cd insert-anything

## 创建和激活运行环境
conda create -n InsertAnything python=3.10 -y
conda activate InsertAnything

## 安装依赖组件
pip install -r requirements.txt
pip install bitsandbytes

## 下载模型
pip install "huggingface_hub[cli]"
huggingface-cli download shiyi0408/InsertAnything --local-dir checkpoints/InsertAnything
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev --local-dir checkpoints/FLUX.1-Fill-dev
huggingface-cli download black-forest-labs/FLUX.1-Redux-dev --local-dir checkpoints/FLUX.1-Redux-dev

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete
python app.py  











 
















