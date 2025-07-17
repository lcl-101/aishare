# Youtube 节目：
## 告别炼丹LoRA！只需一张照片，字节跳动 XVerse 让你的AI角色“一键永驻”！(附本地部署教程)
## https://youtu.be/BnDqaFzYGB0

# 安装指南
## 创建运行环境
conda create -n XVerse python=3.11 -y  
conda activate XVerse  

## 克隆文件
git clone https://github.com/bytedance/XVerse.git  
cd XVerse  

## 安装依赖组件
pip install -r requirements.txt  
pip install flash_attn==2.8.0.post2 --no-build-isolation  

## 下载模型
cd checkpoints  
bash ./download_ckpts.sh  
cd ..  

cp -r /export/demo-softice/models/checkpoints /workspace/XVerse/  

## 启动 WebUI 程序
bash run_demo.sh   



  












 
















