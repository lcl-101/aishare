# Youtube 节目：
## 腾讯王炸开源！Hunyuan-Video-1.5：单卡跑通电影级视频生成？从零部署保姆级教程 + 实测！
## https://youtu.be/2G1uePB2QqA

# 安装指南
## 克隆项目
conda create -n hunyuanvideo-1-5 python=3.10 -y  
conda activate hunyuanvideo-1-5  

## 创建运行环境
conda create -n hunyuanvideo-1-5 python=3.10 -y  
conda activate hunyuanvideo-1-5  

## 安装依赖组件
sed -i 's/torch>=2.6.0/torch==2.6.0/' requirements.txt  
pip install -r requirements.txt  
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
pip install gradio  

## 模型下载
hf download tencent/HunyuanVideo-1.5 --local-dir checkpoints/HunyuanVideo-1.5  
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir checkpoints/HunyuanVideo-1.5/text_encoder/llm  
hf download google/byt5-small --local-dir checkpoints/text_encoder/HunyuanVideo-1.5/byt5-small  
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir checkpoints/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2  
hf download black-forest-labs/FLUX.1-Redux-dev --local-dir checkpoints/HunyuanVideo-1.5/vision_encoder/siglip  

## 推理演示
python app.py      

  












 
















