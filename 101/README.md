# Youtube 节目：
## 字节跳动又开源神器！SuperEdit 本地部署体验，AI 指令P图效果惊艳！ 
## https://youtu.be/s67ZXHhdKoQ

# 安装指南

## 创建和激活运行环境
conda create -n SuperEdit python=3.10 -y    
conda activate SuperEdit  

## 克隆项目
git clone https://github.com/bytedance/SuperEdit.git  
cd SuperEdit  

## 安装依赖组件
bash prepare_env.sh  
pip install gradio  

## 下载模型文件
huggingface-cli download limingcv/SuperEdit_InstructP2P_SD15_BaseInstructDiffusion --local-dir checkpoints/SuperEdit_InstructP2P_SD15_BaseInstructDiffusion  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python3 gradio_demo/gradio_demo.py  










 
















