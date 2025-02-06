### 将课程文件夹当中的工程文件和视频文件复制到指定目录
### 使用 ffmpeg 从视频文件中提取音频文件
### 分析工程文件，读取字幕轨道，提取每一句字幕的开始时间和结束时间
### 依据字幕的开始时间和结束时间，将音频文件分割为多个小段
### 分割的音频文件命名，将从 dataset-v2 文件夹最后一个文件的序号开始
import os
import shutil
import stat
import subprocess
import json
import datetime
import time

course_local_folder = "media"
course_dataset_folder = "dataset/train"
course_dataset_test_folder = "dataset/test"
course_project_file_suffix = '.tscproj'

### 遍历本地课程文件夹，使用 ffmpeg 从视频文件中提取音频文件，音频文件名称与视频文件名称相同
### 提取成功之后，删除视频文件
for root, dirs, files in os.walk(course_local_folder):
    for file in files:
        if os.path.splitext(file)[1] == '.mp4':
            video_full_path = os.path.join(root, file)
            audio_file_name = os.path.splitext(file)[0] + '.wav'
            audio_full_path = os.path.join(root, audio_file_name)
            ffmpeg_exe = 'ffmpeg.exe'
            ffmpeg_execute = subprocess.Popen([ffmpeg_exe, "-i", video_full_path, "-vn", "-y", audio_full_path])
            ffmpeg_code = ffmpeg_execute.wait()
            os.remove(video_full_path)

### 定义文件序号，从 0 开始
txt_file_count = 0

### 遍历 course_dataset_folder 目录，遍历所有 txt 文件，获取 txt 文件的数量
for root, dirs, files in os.walk(course_dataset_folder):
    for file in files:
        if os.path.splitext(file)[1] == '.txt':
            txt_file_count += 1

### 遍历 course_dataset_test_folder 目录，遍历所有 txt 文件，获取 txt 文件的数量
for root, dirs, files in os.walk(course_dataset_test_folder):
    for file in files:
        if os.path.splitext(file)[1] == '.txt':
            txt_file_count += 1


#定义新的列表，用于存储每一句字幕的开始时间和结束时间以及字幕内容
time_line_info = []
### 遍历 course_video_folder 目录，读取工程文件，提取字幕轨道，提取每一句字幕的开始时间和结束时间     
for (dir_path, dir_name, file_name) in os.walk(course_local_folder):
    for name in file_name:
        if name.endswith(course_project_file_suffix):
            file_full_path = os.path.join(dir_path, name)
            tscproj_file = open(file_full_path, encoding='UTF-8')
            data = json.loads(tscproj_file.read())
            tscproj_file.close()
            timeline = data['timeline']
            sceneTrack = timeline['sceneTrack']
            scenes = sceneTrack['scenes']
            csml = scenes[0]['csml']
            tracks = csml['tracks']
            subtitle_track = tracks[1]
            subtitle_media = subtitle_track['medias']
            #获取字幕轨道的长度
            full_length = subtitle_media[0]['duration']
            captionData = subtitle_media[0]['parameters']
            caption = captionData['captionData']
            keyframes = caption['keyframes']
            for keyframe in keyframes:
                #如果是第一个 keyframe，那么 start_time 为 0，end_time 为下一个 keyframe['time']
                #如果不是第一个 keyframe，那么 start_time keyframe['time']，end_time 为下一个 keyframe['time']
                #如果是最后一个 keyframe，那么 start_time keyframe['time']，end_time 为 full_length
                keyframe_index = keyframes.index(keyframe)
                if keyframe_index == 0:
                    start_time = 0
                    end_time = keyframes[keyframe_index + 1]['time']
                elif keyframe_index == len(keyframes) - 1:
                    start_time = keyframe['time']
                    end_time = full_length
                else:
                    start_time = keyframe['time']
                    end_time = keyframes[keyframe_index + 1]['time']
                    
                #依据 start_time 计算时间
                #计算的规则是：将 start_time 除以30取整，得到秒数，余数为毫秒数，最后转换为时间格式
                #例如：start_time 为 51，那么 51 除以 30 取整为 1，余数为 21，那么时间为 00:00:01.21
                #start_time 为 108，那么 108 除以 30 取整为 3，余数为 18，那么时间为 00:00:03.18
                #start_time 为 1828，那么 1828 除以 30 取整为 60，余数为 28，那么时间为 00:01:00.28
                start_time_second = start_time // 30
                start_time_millisecond = start_time % 30

                # start_time_millisecond 除以3，再乘以1000，得到毫秒数，保留三位整数
                start_time_millisecond = round(start_time_millisecond / 3, 2)
                start_time_millisecond = start_time_millisecond * 100
                #合并 start_time_second 和 start_time_millisecond，得到 start_time，格式为00:00:01,700
                start_time = datetime.timedelta(seconds=start_time_second, milliseconds=start_time_millisecond)
                #将 start_time 转换为毫秒
                
    
                #依据 end_time 计算时间
                end_time_second = end_time // 30
                end_time_millisecond = end_time % 30
                #end_time_millisecond 除以3，再乘以1000，得到毫秒数，保留三位整数
                end_time_millisecond = round(end_time_millisecond / 3, 2)
                end_time_millisecond = end_time_millisecond * 100
                #合并 end_time_second 和 end_time_millisecond，得到 end_time，格式为00:00:01,700
                end_time = datetime.timedelta(seconds=end_time_second, milliseconds=end_time_millisecond) 
                text = keyframe['value']['text']
                #将每一句字幕的开始时间和结束时间以及字幕内容存储到 time_line_info 列表中
                time_line_info.append((txt_file_count,start_time, end_time, text,name.split('.')[0]))
                txt_file_count += 1
            os.remove(file_full_path)
            
#基于 start_time 和 end_time信息，使用 ffmpeg 将音频文件分割为多个小段，文件名称为 txt_file_count.wav
#将 text 写入到 txt 文件中，文件名称为 txt_file_count.txt
for time_line in time_line_info:
    txt_file_count = time_line[0] + 1
    start_time = time_line[1]
    end_time = time_line[2]
    text = time_line[3]
    name = time_line[4]
    #定义文件名称，如果txt_file_count 为 1，那么 file_name 为 00000001,如果 txt_file_count 为 100，那么 file_name 为 00000100
    #自动补齐 8 位
    split_file_name = str(txt_file_count).zfill(8)
    split_audio_file_name = split_file_name + '.wav'
    split_txt_file_name = split_file_name + '.txt'
    split_audio_file_path = os.path.join(course_dataset_folder, 'audio', split_audio_file_name)
    split_txt_file_path = os.path.join(course_dataset_folder, 'text', split_txt_file_name)
    #基于 start_time 和 end_time 将音频文件分割为多个小段，文件名称为 txt_file_count.wav
    #将 text 写入到 txt 文件中,文件名称为 txt_file_count.txt
    full_audio_file_path = os.path.join(course_local_folder, name + '.wav')

    
    #使用 ffmpeg，基于 start_time 和 end_time 将音频文件分割为多个小段,并且音频的采样率为 16000
    ffmpeg_exe = 'ffmpeg.exe'
    ffmpeg_execute = subprocess.Popen([ffmpeg_exe, "-i", full_audio_file_path, "-ss", str(start_time), "-to", str(end_time), "-y", split_audio_file_path])
    #ffmpeg_execute = subprocess.Popen([ffmpeg_exe, "-i", full_audio_file_path, "-ss", str(start_time), "-to", str(end_time), "-ar", "16000", "-y", split_audio_file_path])
    ffmpeg_code = ffmpeg_execute.wait()
    if ffmpeg_code != 0:
        print(full_audio_file_path)
        break
    #将 text 写入到 txt 文件中
    txt_file = open(split_txt_file_path, 'w+', encoding='UTF-8')
    txt_file.write(text)
    txt_file.close()
    

#再次遍历 time_line_info 列表，删除 time_line_info 列表中的音频文件
for time_line in time_line_info:
    name = time_line[4]
    audio_file_path = os.path.join(course_local_folder, name + '.wav')
    try:
        os.remove(audio_file_path)
    except:
        pass

#清空 course_local_folder 文件夹当中所有的文件，保留文件夹
for root, dirs, files in os.walk(course_local_folder):
    for file in files:
        file_path = os.path.join(root, file)
        os.remove(file_path)       

#删除 course_local_folder 文件夹
# os.rmdir(course_local_folder)
