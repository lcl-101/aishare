# Youtube 节目：
## AI 学会观察世界了？悟界 Emu 3.5 深度解析与本地部署，亲手体验“预测下一秒”的魔力！
## https://youtu.be/JukhqZ_wa-Y

# 安装指南
## 克隆项目
git clone https://github.com/baaivision/Emu3.5.git  
cd Emu3.5

## 创建运行环境
conda create -n emu3-5 python=3.10 -y  
conda activate emu3-5  

## 安装依赖组件
sed -i -e '1s/>=2.6.0/==2.6.0/' -e '2s/>=0.15.0/==0.21.0/' -e '3s/>=2.0.0/==2.6.0/' requirements.txt  
pip install -r requirements.txt  
pip install gradio  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  

## 模型下载
hf download BAAI/Emu3.5 --local-dir weights/Emu3.5  
hf download BAAI/Emu3.5-Image --local-dir weights/Emu3.5-Image  
hf download BAAI/Emu3.5-VisionTokenizer --local-dir weights/Emu3.5-VisionTokenizer   

## 推理演示
python app.py     

  












 
















