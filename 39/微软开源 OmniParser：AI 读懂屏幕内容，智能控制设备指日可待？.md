# Youtube 节目：
## 微软开源 OmniParser：AI 读懂屏幕内容，智能控制设备指日可待？
## https://youtu.be/Ley7bU-9mZA

# 安装指南


## 克隆项目
cd /home/softice
git clone https://github.com/microsoft/OmniParser.git

## 创建运行环境
cd OmniParser
conda create -n "omni" python==3.12 -y
conda activate omni
pip install -r requirements.txt

## 下载模型
rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence 
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
mv weights/icon_caption weights/icon_caption_florence
## 启动程序
python gradio_demo.py







