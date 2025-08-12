# Youtube 节目：
## 让照片唱歌说话！本地部署阿里EchomimicV3，实测45GB显存+80分钟生成11秒视频，值得玩吗？
## https://youtu.be/dnxGBuHX6Kc

# 安装指南
## 创建运行环境
conda create -n echomimicv3 python=3.10 -y  
conda activate echomimicv3  

##克隆项目
git clone https://github.com/antgroup/echomimic_v3.git echomimicv3  
cd echomimicv3  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型
huggingface-cli download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP --local-dir models/Wan2.1-Fun-V1.1-1.3B-InP  
huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h  
huggingface-cli download BadToBest/EchoMimicV3 --local-dir models/EchoMimicV3  

mv models/BadToBest/transformer/ models/transformer  


## 推理
python app.py   





  












 
















