# Youtube 节目：
## 会“思考”的AI！智谱 GLM-V 本地部署教程，看它如何一步步解开图片谜题，精准理解视频内容！
## https://youtu.be/EBL6aU6CdEI

# 安装指南
## 创建运行环境
conda create -n GLM python=3.10 -y  
conda activate GLM  

## 克隆文件
git clone https://github.com/THUDM/GLM-4.1V-Thinking.git GLM  
cd GLM  

## 安装依赖组件
pip install -r requirements.txt  

## 准备模型文件
huggingface-cli download THUDM/GLM-4.1V-9B-Thinking --local-dir checkpoints/THUDM/GLM-4.1V-9B-Thinking  

mkdir checkpoints
cp -r /export/demo-softice/models/GLM/GLM-4.1V-9B-Thinking/ checkpoints/GLM-4.1V-9B-Thinking  

## 推理
cd inference  
python trans_infer_gradio.py  


一个杯子有多高
请详细描述一下视频的内容
视频的主角是谁
视频当中的鸟叫什么名字

请将这个视频分割为场景，为每个场景提供开始时间、结束时间和详细描述。




  












 
















