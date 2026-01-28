# Youtube 节目：
## 还在等TTS生成？Dia2 开源模型实测：毫秒级响应+双人对话，重塑 AI 语音交互体验！
## https://youtu.be/gQDxIiS4btA

# 安装指南
## 克隆项目
git clone https://github.com/nari-labs/dia2.git   
cd dia2    

## 创建运行环境
conda create -n dia2 python=3.10 -y  
conda activate dia2  

## 安装依赖组件
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio  
pip install -e .  

## 模型下载
hf download nari-labs/Dia2-2B --local-dir checkpoints/Dia2-2B  
hf download kyutai/mimi --local-dir checkpoints/mimi  
hf download openai/whisper-large-v3 --local-dir checkpoints/whisper-large-v3    

## 推理演示
python app.py      

  












 
















