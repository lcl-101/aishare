# Youtube 节目：
## MiniCPM-V-4.5 本地部署与全面实测：看图/识字/读视频/解文档，样样精通！
## https://youtu.be/3ZW3Y-35Kto

# 安装指南
## 创建运行环境
conda create -n minicpmv python=3.10 -y  
conda activate minicpmv   

## 克隆文件
git clone https://github.com/OpenSQZ/MiniCPM-V-CookBook.git minicpmv   
cd minicpmv   

## 安装依赖组件
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128  
pip install -r inference/requirements.txt   

## 下载模型
huggingface-cli download openbmb/MiniCPM-V-4_5 --local-dir checkpoints/MiniCPM-V-4_5   

## 推理
python app.py   

  












 
















