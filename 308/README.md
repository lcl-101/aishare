# Youtube 节目： 
##  显卡里的AI医生！谷歌 MedGemma 1.5 本地部署全指南：读懂X光片、写诊断报告，隐私完全可控！
##  https://youtu.be/vHrxdMDtZ_U
 
# 安装指南 
## 创建运行环境 
conda create -n medgemma python=3.10 -y 
conda activate medgemma 
 
## 克隆项目 
git clone https://github.com/google-health/medgemma.git 
cd medgemma 
 
## 安装依赖组件 
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 
pip install transformers accelerate bitsandbytes pillow gradio openai 
 
## 模型下载 
hf download google/medgemma-1.5-4b-it --local-dir checkpoints/medgemma-1.5-4b-it 
 
## 推理演示 
python app.py 
 