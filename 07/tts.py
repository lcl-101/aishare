
import asyncio
import edge_tts
import datetime

### 获取运行程序的开始时间
start_time = datetime.datetime.now() 

async def main():
   # 定义根目录
    txt_file = 'demo.txt'
    srt_file = "demo.srt"
    chunk_file = 'chunk.txt'
    voice = 'zh-CN-YunjianNeural' #zh-CN-YunjianNeural
    with open(txt_file, 'rb') as f_txt:
        text_content = f_txt.read().decode('utf-8','ignore')
        communicate = edge_tts.Communicate(text_content, voice)
        submaker = edge_tts.SubMaker()
        with open('demo.wav', "wb") as f_audio:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f_audio.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    submaker.feed(chunk)
                    chunk_item_list = []
                    chunk_item_list.append(chunk["offset"])
                    chunk_item_list.append(chunk["duration"])
                    chunk_item_list.append(chunk["text"])
                    with open(chunk_file, 'a', encoding='utf-8') as f_chunk:
                        f_chunk.write(str(chunk_item_list) + '\n')
                    f_chunk.close()

        with open(srt_file, "w", encoding="utf-8") as f_srt:
            f_srt.write(submaker.get_srt())
        f_audio.close()
        f_srt.close()
    f_txt.close()
    

if __name__ == '__main__':
  asyncio.run(main())

### 获取运行程序的结束时间
end_time = datetime.datetime.now()
print(f'程序运行时间: {end_time - start_time}')
