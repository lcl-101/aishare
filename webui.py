#使用 gradio 生成一个页面
#用户可以上传一个音频文件，上传之后，将由 whisper 处理，将文本返回给用户
#以下是 whisper 转写的示例代码


import gradio as gr
import os
import whisper
from whisper import load_model
from whisper.utils import get_writer
import datetime
import torch

def whisper_transcribe(audio_file,model_type,language):
    start_time = datetime.datetime.now()
    if language == "简体中文":
        language_id = 'zh'
    elif language == "英语":
        language_id = 'en'
    ### 定义 model 路径
    model_path = 'model/'+model_type+'.pt'

    ### 定义音视频文件目录
    media_path = 'media'
    ### 定义音频、视频文件后缀
    media_suffix = ['.wav', '.mp3', '.mp4', '.flv', '.avi', '.rmvb', '.mkv', '.wmv', '.rm', '.mov', '.3gp', '.mpeg', '.mpg', '.dat', '.asf', '.flac', '.aac', '.m4a', '.wma', '.ogg', '.ape', '.m4b', '.m4r', '.m4v', '.opus', '.webm', '.amr', '.ts', '.vob', '.wav', '.mp3', '.mp4', '.flv', '.avi', '.rmvb', '.mkv', '.wmv', '.rm', '.mov', '.3gp', '.mpeg', '.mpg', '.dat', '.asf', '.flac', '.aac', '.m4a', '.wma', '.ogg', '.ape', '.m4b', '.m4r', '.m4v', '.opus', '.webm', '.amr', '.ts', '.vob']
    ### 定义设备,如果有GPU则使用GPU,否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    ### 遍历音视频文件目录,并且对于每一个音视频文件生成文字保存在文本文件当中
    for root, dirs, files in os.walk(media_path):
        for file in files:
            media_file_suffix = os.path.splitext(file)[1]
            if media_file_suffix in media_suffix:
                media_file_path = os.path.join(root, file)
                model = whisper.load_model(model_path,device=device,in_memory=True)
                result = model.transcribe(media_file_path,fp16=False,language=language_id)
                txt_writer = get_writer('txt', root)
                srt_writer = get_writer('srt', root)
                txt_writer(result,media_file_path)
                srt_writer(result,media_file_path)
                plain_texts = result['segments']
                final_text = ''
                for plain_text in plain_texts:
                    final_text += plain_text['text']
                    final_text += '\n'
                    

    end_time = datetime.datetime.now()
    run_time = (end_time - start_time).seconds
    return final_text,f"程序运行时间：{run_time} 秒"

#控件定义

demo = gr.Interface(
    fn=whisper_transcribe, 
    inputs=[
        gr.Audio(label="上传音频文件"),
        gr.Radio(["large-v3-turbo","large-v3","medium","small","tiny"],label="选择模型类型"),
        gr.Radio(["简体中文","英语"],label="选择语言")

    ],
    outputs=[
        gr.Textbox(label="转写结果"),
        gr.Textbox(label="运行时间"),
    ],
    title='Whisper 测试',
    description='上传音频文件，选择模型和语言，获取转写结果及程序运行时间。'\
    '\n请提前创建好两个文件夹，Model 和 Media，'\
    '\n请将音频文件放入 Media 文件夹中，'\
    '\n请将模型文件放入 Model 文件夹中。'
)


if __name__ == "__main__":
    demo.launch()



