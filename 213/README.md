# Youtube 节目：
## 视频不再是默片！Character.AI 开源视频模型 OVI 保姆级部署教程，轻松生成带对话和音效的 AI 视频！
## https://youtu.be/L-eNeRBUN-M  

# 安装指南
## 创建运行环境
conda create -n ovi python=3.10 -y  
conda activate ovi  

## 克隆项目
git clone https://github.com/character-ai/Ovi.git  
cd Ovi  

## 安装依赖组件
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install flash_attn --no-build-isolation 

## 下载模型
python download_weights.py   
  
## 准备文件
python app.py   

  












 
















