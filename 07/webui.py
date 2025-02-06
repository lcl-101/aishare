import asyncio
import edge_tts
import datetime
import gradio as gr

async def tts(text, voice='zh-CN-XiaoxiaoNeural'):
    communicate = edge_tts.Communicate(text, voice)
    audio_path = 'demo.wav'
    with open(audio_path, "wb") as f_audio:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f_audio.write(chunk["data"])
    return audio_path

def generate_audio(text):
    start_time = datetime.datetime.now()
    audio_path = asyncio.run(tts(text))
    end_time = datetime.datetime.now()
    print(f'程序运行时间: {end_time - start_time}')
    return audio_path

iface = gr.Interface(
    fn=generate_audio,
    inputs=gr.Textbox(lines=5, label="输入文字"),
    outputs=gr.Audio(label="生成的音频"),
    title="EdgeTTS 测试",
    description="输入文字并生成音频"
)

if __name__ == "__main__":
    iface.launch()