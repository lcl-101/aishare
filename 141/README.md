# Youtube 节目：
## AI开口说相声？开源神器 MOSS-TTSD，一键生成多人对话播客，效果炸裂！
## https://youtu.be/Jy0LQy35Its

# 安装指南
## 创建运行环境
conda create -n TTSD python=3.10 -y  
conda activate TTSD  

## 克隆文件
git clone https://github.com/OpenMOSS/MOSS-TTSD TTSD  
cd TTSD  

## 安装依赖组件
pip install -r requirements.txt  
pip install http://cache.mixazure.com:88/whl/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl  
conda install -c conda-forge libstdcxx-ng -y  

## 下载模型
mkdir -p XY_Tokenizer/weights  
huggingface-cli download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt --local-dir ./XY_Tokenizer/weights/  

cp -r /export/demo-softice/models/XY_Tokenizer/ /workspace/TTSD/  

## 启动 WebUI 程序
python gradio_demo.py  



  












 
















