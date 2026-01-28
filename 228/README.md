# Youtube 节目：
## 给声音加点“戏”！能克隆任何音色、一句话改变情绪的AI音频编辑器 Step-audio-editx 来了！
## https://youtu.be/kj-msOp9qXs

# 安装指南
## 克隆项目
git clone https://github.com/stepfun-ai/Step-Audio-EditX.git  
cd Step-Audio-EditX  

## 创建运行环境
conda create -n stepaudioedit python=3.10 -y  
conda activate stepaudioedit    

## 安装依赖组件
pip install -r requirements.txt  

## 模型下载
hf download stepfun-ai/Step-Audio-EditX --local-dir checkpoints/Step-Audio-EditX  
hf download stepfun-ai/Step-Audio-Tokenizer --local-dir checkpoints/Step-Audio-Tokenizer     

## 推理演示
python app.py --model-path checkpoints/ --model-source local  

  












 
















