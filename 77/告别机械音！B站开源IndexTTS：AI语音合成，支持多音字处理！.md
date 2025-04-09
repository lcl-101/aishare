# Youtube 节目：
## 告别机械音！B站开源IndexTTS：AI语音合成，支持多音字处理！
## https://youtu.be/v-RhAxcteOU

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt-get install ffmpeg  

## 克隆项目
git clone https://github.com/index-tts/index-tts.git  
cd index-tts  

## 创建和激活运行环境
conda create -n IndexTTS python=3.10 -y  
conda activate IndexTTS  

## 安装依赖
pip install -r requirements.txt  

## 下载模型
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_discriminator.pth -P checkpoints  
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_generator.pth -P checkpoints  
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bpe.model -P checkpoints  
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/dvae.pth -P checkpoints  
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/gpt.pth -P checkpoints  
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/unigram_12000.vocab -P checkpoints  

## 推理
find . -name "*:Zone.Identifier" -type f -delete   
python webui.py  

不用说，人们在陆羽之前便已开始饮用茶。饮茶的习俗非常古老，甚至无法估测其出现的时间。只是茶字很可能是到了陆羽的时代才开始使用的，此前使用的是荼、茗、荈、槚等字。  
麻烦的是，若据《说文解字》，荼的本意是苦菜。很可能的情况是：作为饮食用的植物，茶远较苦菜出现得晚，便临时借用了荼作为名字。也就是说，荼是文字上的借用。因此，不能因为《诗经》中出现有荼字，便可断言在孔子以前就有人开始喝茶。《诗经》中的用例是荼毒，意指难以忍受的残酷，决非是指茶。  
如果口干了，饮水即可，而饮茶除了解渴外还有别的目的。与饮水不同，饮茶要花很大的工夫。这可从中窥见文化的萌芽。待陆羽将之体系化以后，饮茶遂脱离了文化的萌芽期，渐渐迎来了成熟的时期。  

同一条路，和某些人一起走，就长的离谱，和另外一些人走，就短得让人舍不得迈步子。  


 
















