### 获取 train_dataset 文件数量
### 随机选择 10000 个文件，将这 10000 个文件移动到 test_dataset 文件夹
### audio 文件放到 test_dataset/audio 文件夹
### text 文件放到 test_dataset/text 文件夹
### 分割的音频文件命名，将从 dataset-v2 文件夹最后一个文件的序号开始
import os
import shutil
import stat
import subprocess
import json
import datetime
import time
import random


course_local_folder = "media"
train_dataset_folder = "dataset/train"
test_dataset_folder = "dataset/test"
log_list = []

### 定义文件序号，从 0 开始
txt_file_count = 0

### 遍历 course_dataset_folder 目录，遍历所有 txt 文件，获取 txt 文件的数量
for root, dirs, files in os.walk(train_dataset_folder):
    for file in files:
        if os.path.splitext(file)[1] == '.txt':
            txt_file_count += 1

#从 1 到 txt_file_count 之间随机选择 1000 个文件
random_list = random.sample(range(1, txt_file_count), 10000)

### 遍历 random_list，如果不足八位数，则在前面补零
### 拼接出 dataset/train/audio 文件夹下的文件名，同时生成 dataset/train/text 文件夹下的文件名
### 生成 dataset/test/audio 文件夹下的文件名，同时生成 dataset/test/text 文件夹下的文件名
### 将 audio 文件和 text 文件移动到 test_dataset 文件夹
for i in random_list:
    file_count = str(i).zfill(8)
    train_audio_file = os.path.join(train_dataset_folder, 'audio', file_count + '.wav')
    train_text_file = os.path.join(train_dataset_folder, 'text', file_count + '.txt')
    test_audio_file = os.path.join(test_dataset_folder, 'audio', file_count + '.wav')
    test_text_file = os.path.join(test_dataset_folder, 'text', file_count + '.txt')
    shutil.move(train_audio_file, test_audio_file)
    shutil.move(train_text_file, test_text_file)


