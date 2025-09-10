# Youtube 节目：
## 挑战腾讯最强AI画图模型！HunyuanImage本地部署实测，显卡当场“核爆”！
## https://youtu.be/CVSu2PMhLh4

# 安装指南
## 创建运行环境
conda create -y -n HunyuanImage python=3.10  
conda activate HunyuanImage   

## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git  
cd HunyuanImage-2.1  
git lfs pull  

## 安装加速组件
pip install -r requirements.txt  
pip install flash-attn==2.7.3 --no-build-isolation  
pip install gradio  

## 下载模型文件
hf download tencent/HunyuanImage-2.1 --local-dir ./ckpts  
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/text_encoder/llm  
hf download google/byt5-small --local-dir ./ckpts/text_encoder/byt5-small  
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2  

## 启动程序
python app.py     

  












 
















