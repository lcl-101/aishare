# Youtube 节目：
## 显存减半，速度翻倍！开源神器 SpotEdit 本地部署：零训练实现像素级精准修图 🔥
## https://youtu.be/azKqqIzJmYg

# 安装指南
## 克隆项目
git clone https://github.com/Biangbiang0321/SpotEdit   
cd SpotEdit   

## 创建运行环境
conda create -n spotedit python=3.10 -y  
conda activate spotedit 

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio accelerate    

## 模型下载
hf download Qwen/Qwen-Image-2512 --local-dir checkpoints/Qwen-Image-2512  
hf download lightx2v/Qwen-Image-2512-Lightning --local-dir checkpoints/Qwen-Image-2512-Lightning  

## 推理演示
python app.py        

  












 
















