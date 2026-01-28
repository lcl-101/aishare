# Youtube 节目：
## AI界的理科学霸！阿里开源Ovis，性能超越GPT-4o？原生高分辨率，看清蚂蚁大小文字，秒解数理化难题！(附本地部署教程)
## https://youtu.be/Nzq02mzWtoE

# 安装指南
## 克隆项目
git clone https://github.com/AIDC-AI/Ovis  
cd Ovis  

## 创建运行环境
conda create -n ovis python=3.10 -y  
conda activate ovis  

## 安装依赖组件
pip install -r requirements.txt  
pip install -e .  
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash-attn==2.7.4.post1 --no-build-isolation  

## 模型下载
hf download AIDC-AI/Ovis2.5-9B --local-dir checkpoints/Ovis2.5-9B  

## 推理演示
python app.py      

  












 
















