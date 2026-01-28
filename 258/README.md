# Youtube 节目：
## One-to-All-Animation，无需裁剪、无需对齐！这款 AI 动画神器彻底解放双手，从零开始本地部署指南
## https://youtu.be/rBnUsQDa5rk

# 安装指南
## 克隆项目
git clone https://github.com/ssj9596/One-to-All-Animation.git  
cd One-to-All-Animation   

## 创建运行环境
conda create -n one-to-all python=3.10 -y  
conda activate one-to-all   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
pip install -r requirements.txt  
pip install gradio  

## 模型下载
cd ./pretrained_models  
python download_pretrained_models.py  
cd ..  
cd ./checkpoints  
bash download_checkpoints.py  



Tip: Edit the script and uncomment the specific models you want to download.  
1.3B_1: Best performance on video benchmark among 1.3B models (paper results).  
1.3B_2: Further trained on v1 with large camera movement data and increased image ratio. Better for dynamic video generation. Best on image benchmark (paper ####results).  
14B: Best overall performance among 14B models (paper results).  

## 推理演示
python app.py        

  












 
















