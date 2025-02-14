### 基于 chunk 和 final.txt 文件，生成 srt 文件
### 对于书名，添加《》
### 对于人名，添加 · 符号

import datetime
import os
import math
import re

### 获取运行程序的开始时间
start_time = datetime.datetime.now() 

#定义根目录
txt_file = 'demo.txt'
chunk_file = 'chunk.txt'
srt_file = "demo.srt"

### 读取文本文件内容，并做分割处理
with open(txt_file, "r", encoding="utf-8") as f_txt:
    ### 读取文本文件的内容
    txt = f_txt.read()
    ### 定义标点替换列表和分割列表
    replace_list = []
    replace_list.append(["\n",""])
    
    replace_list.append(["《",""])
    replace_list.append(["》",""])
    replace_list.append([",", "，"])
    
    replace_list.append([";", "；"])
    replace_list.append(["?", "？"])
    replace_list.append(["!", "！"])
    
    replace_list.append(["”", ""])
    replace_list.append(["“", ""])
    replace_list.append(["·", ""])
    replace_list.append(["、", "，"])
    
    replace_list.append(["（", "，"])
    replace_list.append(["）", "，"])
    replace_list.append(['"', ""])
    replace_list.append(["……", "。"])
    replace_list.append(["‘", ""])
    replace_list.append(["’", ""])
    replace_list.append(["“", ""])
    replace_list.append(["”", ""])
    replace_list.append(["(", "，"])
    replace_list.append(["•", ""])
    replace_list.append(["〖", ""])
    replace_list.append(["〗", ""])
    replace_list.append(["-", ""])
    replace_list.append(["－", ""])
    replace_list.append(["⋯⋯", "，"])
    replace_list.append(["──", "，"])
    replace_list.append(["～", "，"])
    replace_list.append(["…", "，"])
    ### 替换标点符号
    for item in replace_list:
        txt = txt.replace(item[0], item[1])


    ### 定义分割符号
    txt_list = txt.split("。")
    split_list = []
    split_list.append("，")
    split_list.append("？")
    split_list.append("！")
    split_list.append("；")
    split_list.append("：")
    split_list.append("—")


    ### 按照分割符号进行分割
    for item in split_list:
        temp_list = []
        for i in range(0, len(txt_list)):
            txt_list[i].split(item)
            for j in range(0, len(txt_list[i].split(item))):
                temp_list.append(txt_list[i].split(item)[j])
        txt_list = temp_list


    ### 在 txt 列表当中，对于为空的元素，进行删除
    for i in range(len(txt_list)-1,-1,-1):
        if txt_list[i] == "":
            txt_list.pop(i)

    # 对于 txt_list 当中的元素，进行处理，删除\ufeff字符
    for i in range(len(txt_list)):
        txt_list[i] = txt_list[i].replace("\ufeff", "")


    chunk_list = []
    ### 读取 chunk 文件内容，并存储到列当中
    with open(chunk_file, "r", encoding="utf-8") as f_chunk:
        for line in f_chunk:
            chunk_string = line
            #分割为三个元素的列表
            chunk_string_list = chunk_string.split(",")
            #删除 chunk_string_list[0] 的第一个字符[
            chunk_string_list[0] = chunk_string_list[0].replace("[", "")
            #删除 chunk_string_list[2] 的最后一个字符]
            chunk_string_list[2] = chunk_string_list[2].replace("]", "")
            # 删除 chunk_string_list[2] 的空格之后，再删除第一个字符和最后一个字符
            chunk_string_list[2] = chunk_string_list[2].strip()
            chunk_string_list[2] = chunk_string_list[2][1:-1]
            #替换 &amp; 为 &
            chunk_string_list[2] = chunk_string_list[2].replace("&amp;", "&")
            chunk_list.append(chunk_string_list)
    #对于 chunk_list 当中的元素，进行处理，删除\ufeff字符
    for i in range(len(chunk_list)):
        chunk_list[i][0] = chunk_list[i][0].replace("\ufeff", "")
        chunk_list[i][1] = chunk_list[i][1].replace("\ufeff", "")
        #chunk_list[i][2] = chunk_list[i][2].replace("\ufeff", "")
        


    srt_list = []
    for i in range(len(txt_list)):
        j = 0
        txt_str = txt_list[i]
        if txt_str == "":
            break
        if j >= len(chunk_list):
            break
        chunk_str = chunk_list[j][2]
        srt_start_time = chunk_list[j][0]
        #print(chunk_str)
        while chunk_str != txt_str:
            #j += 1
            #chunk_str += chunk_list[j][2]

            #判断 txt_str 当中是否包含 - 符号，如果包含，那么记录 - 符号所在的位置
            #并且在 chunk_str 当中，添加上 - 符号
            #pre_word = ""
            j += 1
            temp_str = chunk_list[j][2]
            
            
            symbol_str = "、"
            if symbol_str in txt_str:
                index = txt_str.index(symbol_str) #-
                pre_word = txt_str[index-1]
                
                if pre_word in temp_str:
                    temp_str = temp_str + symbol_str
            
            #删除 chunk_list[j][2] 的最后一个字符
            #chunk_list[j][2] = chunk_list[j][2][:-1]
            chunk_str += temp_str

            


        srt_end_time = int(chunk_list[j][0]) + int(chunk_list[j][1])
        srt_str = chunk_str
        srt_item_list = [srt_start_time, srt_end_time, srt_str]
        srt_list.append(srt_item_list)
        del chunk_list[0:j+1]

    for i in range(len(srt_list)):
        srt_start_time = srt_list[i][0]
        srt_start_time = srt_start_time.replace(" ", "")
        srt_start_time = srt_start_time.replace("\ufeff", "")

        srt_end_time = srt_list[i][1]

        srt_text = srt_list[i][2]

        srt_start_time = int(srt_start_time)
        srt_end_time = int(srt_end_time)

        hour = math.floor(srt_start_time / 10**7 / 3600)
        minute = math.floor((srt_start_time / 10**7 / 60) % 60)
        seconds = (srt_start_time / 10**7) % 60
        srt_start_time = f'{hour:02d}:{minute:02d}:{seconds:06.3f}'

        hour = math.floor(srt_end_time / 10**7 / 3600)
        minute = math.floor((srt_end_time / 10**7 / 60) % 60)
        seconds = (srt_end_time / 10**7) % 60
        srt_end_time = f'{hour:02d}:{minute:02d}:{seconds:06.3f}'

        with open(srt_file, "a", encoding="utf-8") as f_srt:
            f_srt.write(f'{i+1}\n')
            f_srt.write(f'{srt_start_time} --> {srt_end_time}\n')
            f_srt.write(f'{srt_text}\n')
            f_srt.write('\n')
        #关闭文件
        f_txt.close()
        f_chunk.close()
        f_srt.close()


  
### 获取运行程序的结束时间
end_time = datetime.datetime.now()
print(f'程序运行时间: {end_time - start_time}')
