# Youtube 节目：
## AI音乐制作人必备！本地部署SongPrep：精准提取歌词、自动划分主副歌 (保姆级教程)
## https://youtu.be/btm0H3RTwww

# 安装指南
## 克隆项目
git clone https://github.com/tencent-ailab/SongPrep.git  
cd SongPrep  

## 创建运行环境
conda create -n songprep python=3.10 -y  
conda activate songprep   

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio omegaconf  

## 模型下载
hf download waytan22/SongPrep-7B --local-dir checkpoints/SongPrep-7B  

## 推理演示
python app.py    

  












 
















