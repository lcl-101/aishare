# Youtube 节目：
## 想要懂“黑话”的私人翻译官？教你本地部署混元 MT 1.5：支持术语干预、全量微调！
## https://youtu.be/E4CrNmMy6sw

# 安装指南
## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HY-MT.git  
cd HY-MT  

## 创建运行环境
conda create -n hy-mt python=3.10 -y  
conda activate hy-mt  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers==4.56.0 accelerate gradio  

## 模型下载
hf download tencent/HY-MT1.5-1.8B --local-dir checkpoints/HY-MT1.5-1.8B  

## 推理演示
python app.py        

  












 
















