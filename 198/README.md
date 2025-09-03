# Youtube 节目：
## Stand-In 项目实战：本地部署微信团队AI视频生成工具，实现人物/动作/风格精准控制
## https://youtu.be/92tAm1PQUU0

# 安装指南
## 创建运行环境
conda create -n Stand-In python=3.11 -y  
conda activate Stand-In  

## 克隆文件
git clone https://github.com/WeChatCV/Stand-In.git  
cd Stand-In  

## 安装依赖组件
pip install -r requirements.txt  
pip install flash-attn --no-build-isolation  
pip install gradio  

## 下载模型
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir  checkpoints/VACE  
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir  checkpoints/base_model  
huggingface-cli download DIAMONIK7777/antelopev2 --local-dir  checkpoints/antelopev2/models/antelopev2  
huggingface-cli download BowenXue/Stand-In --local-dir checkpoints/Stand-In  

## 推理
python app.py  


  












 
















