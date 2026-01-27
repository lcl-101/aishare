# Youtube 节目：
## 碾压传统TTS？VoxCPM 1.5 强力发布：支持情感微调的开源语音克隆神器！
## https://youtu.be/34UqR3ZcwsQ

# 安装指南
## 克隆项目
git clone https://github.com/OpenBMB/VoxCPM  
cd VoxCPM  

## 创建运行环境
conda create -n VoxCPM python=3.10 -y  
conda activate VoxCPM  

## 安装依赖组件
pip install -e .  
pip install torchcodec  

## 模型下载
hf download openbmb/VoxCPM1.5 --local-dir checkpoints/VoxCPM1.5  
modelscope download --model iic/SenseVoiceSmall --local_dir checkpoints/SenseVoiceSmall  
modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir checkpoints/speech_zipenhancer_ans_multiloss_16k_base  

## 推理演示
python app.py        

  












 
















