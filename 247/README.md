# Youtube 节目：
## AI音乐界的ChatGPT？YuE(乐)开源模型实测：支持中英日韩，一键生成专业级全曲！
## https://youtu.be/grCyOY6keiM

# 安装指南
## 创建项目目录
git clone https://github.com/multimodal-art-projection/YuE.git  
cd YuE/inference/  
git clone https://huggingface.co/m-a-p/xcodec_mini_infer  
cd ..  

## 创建运行环境
conda create -n yue python=3.10 -y  
conda activate yue   

## 安装依赖组件
sed -i 's/^torch$/torch==2.6.0/' requirements.txt && sed -i 's/^torchaudio$/torchaudio==2.6.0/' requirements.txt  
pip install -r requirements.txt  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
pip install gradio  

## 模型下载
hf download m-a-p/YuE-s1-7B-anneal-en-cot --local-dir checkpoints/YuE-s1-7B-anneal-en-cot  
hf download m-a-p/YuE-s2-1B-general --local-dir checkpoints/YuE-s2-1B-general  

## 推理演示
python app.py      

  












 
















