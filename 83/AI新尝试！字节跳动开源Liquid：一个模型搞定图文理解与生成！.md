# Youtube 节目：
## AI新尝试！字节跳动开源Liquid：一个模型搞定图文理解与生成！
## https://youtu.be/Jq6cRh1k2aM

# 安装指南

## 克隆项目
git clone https://github.com/FoundationVision/Liquid.git
cd Liquid

## 创建和激活运行环境
conda create -n Liquid python=3.10 -y
conda activate Liquid

## 安装依赖
pip install -r requirements.txt
pip install pydantic==2.10.6  

## 下载模型
cd evaluation/
huggingface-cli download Junfeng5/Liquid_V1_7B  --local-dir  checkpoints/Liquid_V1_7B
wget -P chameleon/ https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.ckpt 
wget -P chameleon/ https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.yaml

## 推理
find . -name "*:Zone.Identifier" -type f -delete  
python app.py





 
















