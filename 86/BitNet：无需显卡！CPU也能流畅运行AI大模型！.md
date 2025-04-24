# Youtube 节目：
## BitNet：无需显卡！CPU也能流畅运行AI大模型！
## https://youtu.be/HD6BnWJpwuk

# 安装指南

## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"  

## 克隆项目
git clone --recursive https://github.com/microsoft/BitNet.git  
cd BitNet  

## 创建和激活运行环境
conda create -n bitnet-cpp python=3.9 -y  
conda activate bitnet-cpp  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T  

## 编译程序
sudo apt install cmake -y  
export CC="clang-19"  
export CXX="clang++-19"  
cmake -B build -DGGML_BITNET_ARM_TL1=ON -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19  

python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s  

## 推理
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv  

python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv -n 2048  










 
















