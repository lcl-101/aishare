# Youtube 节目：
## 斩获30项世界冠军！腾讯开源最强翻译模型Hunyuan-MT，免费可商用，保姆级本地部署教程！
## https://youtu.be/Oju2aebaH5U

# 安装指南
## 克隆项目
git clone https://github.com/Tencent-Hunyuan/Hunyuan-MT.git  
cd Hunyuan-MT  

## 创建运行环境
conda create -n hunyuan-mt python=3.10 -y  
conda activate hunyuan-mt  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers==4.56.0 accelerate==0.33.0 flash_attn gradio  

## 模型下载
hf download tencent/Hunyuan-MT-7B --local-dir checkpoints/Hunyuan-MT-7B   

## 推理演示
python app.py      

  












 
















