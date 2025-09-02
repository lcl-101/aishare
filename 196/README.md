# Youtube 节目：
## 腾讯开源翻译大模型 Hunyuan-MT 测评：支持33种语言，本地部署+WebUI保姆级教程！ 
## https://youtu.be/rw0mDzAcaFo

# 安装指南
## 创建运行环境
conda create -n HunyuanMT python=3.10 -y  
conda activate HunyuanMT  

## 克隆文件
git clone https://github.com/Tencent-Hunyuan/Hunyuan-MT.git HunyuanMT  
cd HunyuanMT  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  

## 准备模型文件
huggingface-cli download tencent/Hunyuan-MT-7B --local-dir weights/Hunyuan-MT-7B  

mkdir checkpoints
cp -r /workspace/checkpoints/Hunyuan-MT-7B/ checkpoints/Hunyuan-MT-7B/  

## 推理
python app.py  

我说一句你说一车啊  

离谱她妈给离谱开门，离谱到家了  

雨女无瓜  

Their relationship is a total situationship.   


  












 
















