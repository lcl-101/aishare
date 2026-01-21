"""
HeartMuLa 音乐生成 Web 应用
基于 Gradio 构建的交互式音乐生成界面
"""

import os
import sys
import tempfile
import shutil

# 处理模型路径结构 - 创建符号链接使路径兼容
MODEL_BASE_PATH = "./checkpoints"
HEARTMULAGEN_PATH = os.path.join(MODEL_BASE_PATH, "HeartMuLaGen")

# 如果 tokenizer.json 在 HeartMuLaGen 子目录中，创建符号链接到 checkpoints 根目录
tokenizer_src = os.path.join(HEARTMULAGEN_PATH, "tokenizer.json")
tokenizer_dst = os.path.join(MODEL_BASE_PATH, "tokenizer.json")
if os.path.exists(tokenizer_src) and not os.path.exists(tokenizer_dst):
    os.symlink(os.path.abspath(tokenizer_src), tokenizer_dst)

gen_config_src = os.path.join(HEARTMULAGEN_PATH, "gen_config.json")
gen_config_dst = os.path.join(MODEL_BASE_PATH, "gen_config.json")
if os.path.exists(gen_config_src) and not os.path.exists(gen_config_dst):
    os.symlink(os.path.abspath(gen_config_src), gen_config_dst)

import gradio as gr
import torch
from heartlib import HeartMuLaGenPipeline

# 全局变量
pipe = None
MODEL_VERSION = "3B"

# 示例歌词
EXAMPLE_LYRICS = """[Intro]

[Verse]
The sun creeps in across the floor
I hear the traffic outside the door
The coffee pot begins to hiss
It is another morning just like this

[Prechorus]
The world keeps spinning round and round
Feet are planted on the ground
I find my rhythm in the sound

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat
It is the ordinary magic that we meet

[Verse]
The hours tick deeply into noon
Chasing shadows,chasing the moon
Work is done and the lights go low
Watching the city start to glow

[Bridge]
It is not always easy,not always bright
Sometimes we wrestle with the night
But we make it to the morning light

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat

[Outro]
Just another day
Every single day"""

# 示例标签
EXAMPLE_TAGS = "piano,happy"

# 中文示例歌词
EXAMPLE_LYRICS_CN = """[Verse]
闭上双眼让喧嚣都停摆
窗外的风轻轻穿过那片海
心里的尘埃慢慢落下来
此刻只有呼吸还在

[Prechorus]
不需要寻找繁复的答案
只需要点亮微弱的烛光
某种力量在悄然生长

[Chorus]
在一片静默里我听见应许
像晨曦温柔洒满了大地
信念是无声却坚定的言语
灵魂终于寻回了栖息地

[Verse]
纯粹的弦音在空气中摇曳
指尖下流淌过岁月的更迭
放下了那些沉重的纠结
在这瞬间与自己和解

[Chorus]
在一片静默里我听见应许
像晨曦温柔洒满了大地
不用去怀疑未知的结局
每一步都走在光里

[Outro]
如风过境如此宁静"""

# 中文示例标签
EXAMPLE_TAGS_CN = "meditation,faith,acoustic,peaceful"


def load_model():
    """加载模型"""
    global pipe
    if pipe is None:
        print("正在加载模型，请稍候...")
        pipe = HeartMuLaGenPipeline.from_pretrained(
            MODEL_BASE_PATH,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            version=MODEL_VERSION,
        )
        print("模型加载完成！")
    return pipe


# 启动时加载模型
print("🚀 正在初始化 HeartMuLa 模型...")
load_model()
print("✅ 模型加载完成，正在启动 Web 界面...")


def generate_music(
    lyrics: str,
    tags: str,
    max_audio_length_sec: int,
    topk: int,
    temperature: float,
    cfg_scale: float,
    progress=gr.Progress()
):
    """生成音乐的主函数"""
    
    if not lyrics.strip():
        raise gr.Error("请输入歌词内容！")
    
    if not tags.strip():
        raise gr.Error("请输入音乐标签！")
    
    progress(0.1, desc="正在准备生成...")
    model = load_model()
    
    progress(0.2, desc="正在生成音乐，请耐心等待...")
    
    # 创建临时文件保存输出
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        save_path = tmp_file.name
    
    max_audio_length_ms = max_audio_length_sec * 1000
    
    try:
        with torch.no_grad():
            model(
                {
                    "lyrics": lyrics,
                    "tags": tags,
                },
                max_audio_length_ms=max_audio_length_ms,
                save_path=save_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )
        
        progress(1.0, desc="生成完成！")
        return save_path
        
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise gr.Error(f"生成失败：{str(e)}")


def use_example_en():
    """使用英文示例内容"""
    return EXAMPLE_LYRICS, EXAMPLE_TAGS


def use_example_cn():
    """使用中文示例内容"""
    return EXAMPLE_LYRICS_CN, EXAMPLE_TAGS_CN


# 创建 Gradio 界面
with gr.Blocks(
    title="HeartMuLa 音乐生成",
    theme=gr.themes.Soft(),
    css="""
    .youtube-banner {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .youtube-banner a {
        color: white !important;
        text-decoration: none;
        font-size: 18px;
        font-weight: bold;
    }
    .youtube-banner a:hover {
        text-decoration: underline;
    }
    """
) as demo:
    
    # YouTube 频道信息横幅
    gr.HTML("""
        <div class="youtube-banner">
            <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">
                🎬 欢迎订阅我的 YouTube 频道：AI 技术分享频道
            </a>
        </div>
    """)
    
    gr.Markdown("""
    # 🎵 HeartMuLa 音乐生成系统
    
    基于 HeartMuLa 模型的 AI 音乐生成工具。输入歌词和音乐风格标签，即可生成独特的音乐作品。
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 输入参数")
            
            lyrics_input = gr.Textbox(
                label="歌词内容",
                placeholder="请输入歌词，支持 [Intro]、[Verse]、[Chorus] 等标记...",
                lines=15,
                max_lines=30,
            )
            
            tags_input = gr.Textbox(
                label="音乐风格标签",
                placeholder="例如：piano,happy,romantic（多个标签用英文逗号分隔）",
                lines=2,
            )
            
            with gr.Row():
                example_btn_en = gr.Button("📋 English Example", variant="secondary")
                example_btn_cn = gr.Button("📋 中文示例", variant="secondary")
            
            gr.Markdown("### ⚙️ 生成参数")
            
            with gr.Row():
                max_length_slider = gr.Slider(
                    minimum=30,
                    maximum=240,
                    value=120,
                    step=10,
                    label="最大音频时长（秒）",
                )
            
            with gr.Row():
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-K 采样参数",
                    info="控制生成的多样性，值越大越多样"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="温度参数",
                    info="控制生成的随机性，值越大越随机"
                )
            
            cfg_scale_slider = gr.Slider(
                minimum=1.0,
                maximum=5.0,
                value=1.5,
                step=0.1,
                label="CFG 引导强度",
                info="Classifier-Free Guidance 强度"
            )
            
            generate_btn = gr.Button("🎶 开始生成音乐", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎧 生成结果")
            
            audio_output = gr.Audio(
                label="生成的音乐",
                type="filepath",
                interactive=False,
            )
            
            gr.Markdown("""
            ### 📖 使用说明
            
            1. **歌词格式**：建议使用歌曲结构标记，如 `[Intro]`、`[Verse]`、`[Chorus]`、`[Bridge]`、`[Outro]` 等
            2. **标签格式**：多个标签用英文逗号分隔，不要有空格，例如：`piano,happy,romantic`
            3. **生成时间**：根据音频时长，生成可能需要几分钟，请耐心等待
            4. **推荐配置**：首次使用建议先用默认参数尝试
            
            ### 🏷️ 常用标签示例
            
            **乐器**：piano, guitar, violin, synthesizer, drums, bass  
            **情绪**：happy, sad, romantic, energetic, calm, melancholic  
            **风格**：pop, rock, jazz, classical, electronic, folk  
            **场景**：wedding, party, meditation, workout, study
            """)
    
    # 事件绑定
    example_btn_en.click(
        fn=use_example_en,
        outputs=[lyrics_input, tags_input]
    )
    
    example_btn_cn.click(
        fn=use_example_cn,
        outputs=[lyrics_input, tags_input]
    )
    
    generate_btn.click(
        fn=generate_music,
        inputs=[
            lyrics_input,
            tags_input,
            max_length_slider,
            topk_slider,
            temperature_slider,
            cfg_scale_slider,
        ],
        outputs=audio_output,
    )
    
    gr.Markdown("""
    ---
    *基于 [HeartMuLa](https://github.com/HeartMuLa/heartlib) 开源项目 | 仅供非商业研究和教育用途*
    """)


if __name__ == "__main__":
    # 预加载模型（可选，取消注释以在启动时加载）
    # load_model()
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
