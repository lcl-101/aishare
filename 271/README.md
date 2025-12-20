# Youtube 节目：
## AI 生成的视频也能有“记忆”？腾讯混元 World 1.5 开源部署全流程，实测键盘控制镜头！
## https://youtu.be/hkIdtRgDM2U

# 安装指南
## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git  
cd HY-WorldPlay  

## 创建运行环境
conda create --name worldplay python=3.10 -y  
conda activate worldplay  

## 安装依赖组件 
sed -i '/^torch>=2.6.0$/d; /^torchaudio==2.6.0$/d' /workspace/HY-WorldPlay/requirements.txt  
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
pip install -r requirements.txt  
pip install gradio  

## 模型下载
hf download tencent/HY-WorldPlay --local-dir checkpoints/HY-WorldPlay  
hf download tencent/HunyuanVideo-1.5 --local-dir checkpoints/HunyuanVideo-1.5  
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir checkpoints/HunyuanVideo-1.5/text_encoder/llm  
hf download google/byt5-small --local-dir checkpoints/text_encoder/HunyuanVideo-1.5/text_encoder/byt5-small  
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir checkpoints/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2  
hf download black-forest-labs/FLUX.1-Redux-dev --local-dir checkpoints/HunyuanVideo-1.5/vision_encoder/siglip  

## 推理演示
python app.py        

  












 
















