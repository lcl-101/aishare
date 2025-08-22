# Youtube 节目：
## 腾讯AI动画神器ToonComposer！仅需几张图，秒生流畅动画！本地部署实测：66GB显存劝退？保姆级教程来了！
## https://youtu.be/b9jkaaHMfVs

# 安装指南
## 创建和激活运行环境
conda create -n ToonComposer python=3.10 -y  
conda activate ToonComposer  

## 克隆项目
git clone https://github.com/TencentARC/ToonComposer  
cd ToonComposer  


## 安装依赖组件
pip install -r requirements.txt  
pip install flash-attn --no-build-isolation  

## 下载模型文件
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir checkpoints/Wan2.1-I2V-14B-480P  
huggingface-cli download TencentARC/ToonComposer --local-dir checkpoints/ToonComposer  

## 启动程序
python app.py  




  












 
















