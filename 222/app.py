#!/usr/bin/env python3
"""
SongBloom WebUI - 简化版 Gradio 界面
基于官方 infer.py 改编
"""

import os
import sys

# ===================== 配置 =====================
MODEL_NAME = "songbloom_full_240s"
LOCAL_DIR = "./checkpoints/SongBloom_long"

# 性能配置选项：
# 选项1（推荐）: DTYPE="float32", ENABLE_FLASH_ATTN=False - 最稳定，速度较慢
# 选项2（实验）: DTYPE="bfloat16", ENABLE_FLASH_ATTN=True - 速度快，需要测试稳定性
DTYPE = "float32"  
ENABLE_FLASH_ATTN = False  # 设置为 True 可启用 Flash Attention（需要配合 bfloat16）

DEVICE = "cuda:0"
OUTPUT_DIR = "./outputs"
N_SAMPLES_DEFAULT = 1

HOST = "0.0.0.0"
PORT = 7860
# ===============================================

# 根据配置决定是否启用 Flash Attention
if not ENABLE_FLASH_ATTN:
    os.environ['DISABLE_FLASH_ATTN'] = "1"

import json
import torch
import torchaudio
import gradio as gr
from datetime import datetime
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler
from normalize_lyrics import clean_lyrics


def load_config(cfg_file, parent_dir="./"):
    """加载配置文件"""
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: OmegaConf.load(x))
    OmegaConf.register_new_resolver("dynamic_path", lambda x: x.replace("???", parent_dir))
    
    file_cfg = OmegaConf.load(open(cfg_file, 'r'))
    return file_cfg


def download_required_files(local_dir, model_name):
    """下载所需的配置和模型文件"""
    print("检查必需文件...")
    
    # 确定仓库ID
    repo_id_map = {
        "songbloom_full_150s": "CypressYang/SongBloom",
        "songbloom_full_150s_dpo": "CypressYang/SongBloom",
        "songbloom_full_240s": "CypressYang/SongBloom_long",
    }
    main_repo_id = repo_id_map.get(model_name, "CypressYang/SongBloom_long")
    
    # 需要下载的文件列表
    files_to_download = [
        ("stable_audio_1920_vae.json", "CypressYang/SongBloom", "VAE 配置文件"),
        ("autoencoder_music_dsp1920.ckpt", "CypressYang/SongBloom", "VAE 权重文件（较大，可能需要几分钟）"),
        ("vocab_g2p.yaml", "CypressYang/SongBloom", "G2P 词汇表文件"),
    ]
    
    for filename, repo_id, description in files_to_download:
        file_path = os.path.join(local_dir, filename)
        if not os.path.exists(file_path):
            print(f"下载 {description}...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir
                )
                print(f"✓ {description}已下载")
            except Exception as e:
                print(f"⚠️ 下载 {filename} 失败: {e}")
                raise
    
    print("✓ 所有必需文件准备完成")


class SongBloomGenerator:
    """SongBloom 生成器封装"""
    
    def __init__(self, model_name, local_dir, dtype='float32', device='cuda:0'):
        self.model_name = model_name
        self.local_dir = local_dir
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        
        # 下载必需文件（如果需要）
        download_required_files(local_dir, model_name)
        
        # 加载配置
        cfg_path = os.path.join(local_dir, f"{model_name}.yaml")
        print(f"加载配置文件: {cfg_path}")
        self.cfg = load_config(cfg_path, parent_dir=local_dir)
        self.cfg.max_dur = self.cfg.max_dur + 10
        
        # 构建模型
        print(f"加载模型 {model_name}...")
        self.model = SongBloom_Sampler.build_from_trainer(
            self.cfg, strict=False, dtype=self.dtype, device=self.device
        )
        
        # 设置生成参数
        gen_params = dict(self.cfg.inference)
        # 添加 max_frames 参数（关键！控制生成长度）
        # 240s 模型: max_frames = 240 * 25 = 6000
        if 'max_frames' not in gen_params:
            gen_params['max_frames'] = int(self.cfg.max_dur * 25)
            print(f"设置 max_frames = {gen_params['max_frames']} (约 {self.cfg.max_dur} 秒)")
        
        self.model.set_generation_params(**gen_params)
        print("✓ 模型加载完成！")
    
    def generate(self, lyrics, prompt_wav_path, n_samples=1, progress=None):
        """生成歌曲"""
        results = []
        
        # 加载提示音频
        if prompt_wav_path and os.path.exists(prompt_wav_path):
            if progress:
                progress(0.1, "加载提示音频...")
            actual_prompt_path = prompt_wav_path
        else:
            # 使用默认的示例音频
            if progress:
                progress(0.1, "使用默认参考音频...")
            actual_prompt_path = "example/test.wav"
            if not os.path.exists(actual_prompt_path):
                raise FileNotFoundError(f"默认参考音频不存在: {actual_prompt_path}")
        
        # 加载音频文件
        prompt_wav, sr = torchaudio.load(actual_prompt_path)
        if sr != self.model.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, self.model.sample_rate)
        # 转换为单声道，确保是 2D (1, samples)
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True).to(self.dtype)
        # 截取到 10 秒
        prompt_wav = prompt_wav[..., :10 * self.model.sample_rate]
        
        # 生成多个样本
        for i in range(n_samples):
            if progress:
                progress(0.2 + (i * 0.7 / n_samples), f"生成样本 {i+1}/{n_samples}...")
            
            # 生成
            wav = self.model.generate(lyrics, prompt_wav)
            results.append(wav)
        
        if progress:
            progress(1.0, "完成！")
        
        return results


# 全局变量
generator = None


def initialize_model():
    """初始化模型"""
    global generator
    if generator is None:
        print("\n" + "="*50)
        print("初始化 SongBloom 模型...")
        print(f"模型: {MODEL_NAME}")
        print(f"目录: {LOCAL_DIR}")
        print(f"精度: {DTYPE}")
        print(f"设备: {DEVICE}")
        print("="*50 + "\n")
        
        generator = SongBloomGenerator(
            model_name=MODEL_NAME,
            local_dir=LOCAL_DIR,
            dtype=DTYPE,
            device=DEVICE
        )
    return generator


def generate_song(lyrics, prompt_audio, n_samples, progress=gr.Progress()):
    """Gradio 生成函数"""
    try:
        # 确保模型已加载
        gen = initialize_model()
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"webui_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成
        results = gen.generate(lyrics, prompt_audio, n_samples, progress)
        
        # 保存所有生成的音频
        saved_files = []
        for i, wav in enumerate(results):
            output_file = os.path.join(output_dir, f"sample_{i}.flac")
            torchaudio.save(output_file, wav[0].cpu().float(), gen.model.sample_rate)
            saved_files.append(output_file)
        
        # 返回第一个样本用于预览
        if saved_files:
            audio_data, sr = torchaudio.load(saved_files[0])
            preview_audio = (sr, audio_data.numpy().T)
        else:
            preview_audio = None
        
        # 生成信息
        info = {
            "success": True,
            "model": MODEL_NAME,
            "lyrics": lyrics,
            "prompt_audio": prompt_audio,
            "n_samples": n_samples,
            "output_dir": output_dir,
            "files": saved_files,
            "timestamp": timestamp,
            "sample_rate": gen.model.sample_rate
        }
        
        return preview_audio, json.dumps(info, indent=2, ensure_ascii=False)
        
    except Exception as e:
        import traceback
        error_info = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"错误: {e}")
        traceback.print_exc()
        return None, json.dumps(error_info, indent=2, ensure_ascii=False)


# 示例歌词 (240s 模型格式)
EXAMPLE_LYRICS_240S = """[intro] [intro] , [verse] 在寂静的夜里.旋律悄然苏醒.像花瓣一样.轻轻绽放在风里.每一个音符.都是心跳的痕迹.带我走向未知的奇迹.无数色彩在空气中涌动.灵感在指尖跳跃成梦 , [chorus] SongBloom.让歌声绽放天空.点亮世界最温柔的心动.SongBloom.跨越时间的河流.用旋律把梦想托起.永不落空 , [inst] [inst] [inst] , [verse] 心里的秘密.化作和声交织.像春天的花海.彼此回应呼吸.世界在倾听.故事正在继续.把未来写进音乐里.无数色彩在空气中涌动.灵感在指尖跳跃成梦 , [chorus] SongBloom.让歌声绽放天空.点亮世界最温柔的心动.SongBloom.跨越时间的河流.用旋律把梦想托起.永不落空 ,  [inst] [inst] , [bridge] 即使黑夜再长.星光依然明亮.有歌声相伴.就有无限希望 , [chorus] SongBloom.让灵魂随风舞动.把每一颗心点亮成宇宙.SongBloom.让未来一起合奏.让世界听见我们的梦.永远相拥 , [outro] [outro] [outro] [outro]"""


def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="SongBloom WebUI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎵 SongBloom WebUI")
        gr.Markdown(f"基于 **{MODEL_NAME}** 模型的歌曲生成界面 - 最大时长 240 秒（4分钟）")
        gr.Markdown(f"**当前配置**: {DTYPE} 精度, Flash Attention: {'✅ 启用' if ENABLE_FLASH_ATTN else '❌ 禁用'}")
        
        # 标签说明
        with gr.Accordion("🏷️ 歌词标签说明", open=True):
            gr.Markdown("""
### 结构标签（需要重复来控制时长，1个标签 ≈ 5秒）

| 标签 | 用途 | 示例 | 说明 |
|------|------|------|------|
| `[intro]` | 前奏 | `[intro] [intro]` | 器乐开场，重复2次约10秒 |
| `[verse]` | 主歌 | `[verse] 歌词内容` | 后面跟歌词文本 |
| `[chorus]` | 副歌 | `[chorus] 歌词内容` | 后面跟歌词文本，通常是高潮部分 |
| `[inst]` | 间奏 | `[inst] [inst] [inst]` | 纯器乐段落，重复3次约15秒 |
| `[bridge]` | 桥段 | `[bridge] 歌词内容` | 过渡段落，后面跟歌词文本 |
| `[outro]` | 尾奏 | `[outro] [outro] [outro] [outro]` | 歌曲结尾，重复4次约20秒 |

### 格式规则
1. **用逗号 `,` 分隔不同段落**（重要！）
2. **歌词内用句号 `.` 分隔句子**
3. **器乐段重复标签**控制时长（不加歌词文本）
4. **演唱段只写一次标签**，后面跟歌词

### 完整示例
```
[intro] [intro] , [verse] 第一句歌词.第二句歌词 , [chorus] 副歌第一句.副歌第二句 , [inst] [inst] [inst] , [verse] 第二段主歌 , [chorus] 副歌重复 , [outro] [outro]
```
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入")
                
                lyrics_input = gr.Textbox(
                    label="歌词内容",
                    lines=8,
                    max_lines=15,
                    value=EXAMPLE_LYRICS_240S,
                    placeholder="按照上方标签说明输入歌词...",
                    show_copy_button=True
                )
                
                prompt_audio_input = gr.Audio(
                    label="提示音频（可选，上传10秒音频作为风格参考，不上传则使用默认）",
                    type="filepath"
                )
                
                n_samples_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=N_SAMPLES_DEFAULT,
                    label="生成样本数（生成多个版本供选择，每个版本略有不同）"
                )
                
                generate_btn = gr.Button("🎵 开始生成", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎧 输出")
                
                output_audio = gr.Audio(
                    label="生成的音频",
                    type="numpy",
                    show_download_button=True
                )
                
                output_info = gr.JSON(label="生成详情")
        
        # 使用说明
        with gr.Accordion("📖 详细说明", open=False):
            gr.Markdown(f"""
### 模型信息
- **模型名称**: {MODEL_NAME}
- **最大时长**: 240 秒（4分钟）
- **采样率**: 48kHz
- **精度**: {DTYPE}
- **Flash Attention**: {'启用' if ENABLE_FLASH_ATTN else '禁用'}

### 性能优化建议
- **当前配置（稳定）**: float32 + 禁用 Flash Attention - 速度较慢但最稳定
- **可选配置（快速）**: 在代码顶部修改 `DTYPE="bfloat16"` 和 `ENABLE_FLASH_ATTN=True` - 速度提升约 2-3 倍，但可能不够稳定

### 提示音频
- 上传一个 **10秒** 的音频文件作为风格参考
- 支持格式：WAV, FLAC, MP3 等
- 如果不上传，将使用默认参考音频 (example/test.wav)

### 输出
- 生成的音频保存在 `{OUTPUT_DIR}/` 目录
- 每次生成创建一个带时间戳的子目录
- 如果生成多个样本，文件名为 `sample_0.flac`, `sample_1.flac` 等
- 界面预览显示第一个样本

### 性能参考
- 生成时间：约 **10-20 分钟**（取决于歌词长度和配置）
  - float32 + 无 Flash Attention: ~15-20 分钟
  - bfloat16 + Flash Attention: ~5-10 分钟（需手动修改配置）
- 显存占用：约 20-30 GB
            """)
        
        # 示例
        with gr.Accordion("� 预设示例", open=False):
            gr.Markdown("**点击示例可快速填充输入框：**")
            gr.Examples(
                examples=[
                    # 示例1：中文流行歌曲 - 想见你
                    [
                        "[intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] , [verse] 风轻轻吹过古道.岁月在墙上刻下记号.梦中你笑得多甜.醒来却只剩下寂寥.繁花似锦的春天.少了你的色彩也失了妖娆 , [chorus] 想见你.在晨曦中.在月光下.每个瞬间都渴望.没有你.星辰也黯淡.花香也无味.只剩下思念的煎熬.想见你.穿越千山万水.只为那一瞥.你的容颜 , [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] , [verse] 月儿弯弯照九州.你是否也在仰望同一片天空.灯火阑珊处.我寻觅你的影踪.回忆如波光粼粼.荡漾在心湖的每个角落 , [chorus] 想见你.在晨曦中.在月光下.每个瞬间都渴望.没有你.星辰也黯淡.花香也无味.只剩下思念的煎熬.想见你.穿越千山万水.只为那一瞥.你的容颜 , [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro]",
                        "example/test.wav",
                        1
                    ],
                    # 示例2：英文歌曲 - Run with me
                    [
                        "[intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] , [verse] City lights flicker through the car window. Dreams pass fast where the lost ones go. Neon signs echo stories untold. I chase shadows while the night grows cold , [chorus] Run with me down the empty street. Where silence and heartbeat always meet. Every breath. a whispered vow. We are forever. here and now , [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] , [verse] Footsteps loud in the tunnel of time. Regret and hope in a crooked rhyme. You held my hand when I slipped through the dark. Lit a match and you became my spark , [bridge] We were nothing and everything too. Lost in a moment. found in the view. Of all we broke and still survived. Somehow the flame stayed alive , [chorus] Run with me down the empty street. Where silence and heartbeat always meet. Every breath. a whispered vow. We are forever. here and now , [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro]",
                        "example/test.wav",
                        1
                    ],
                    # 示例3：SongBloom主题歌
                    [
                        "[intro] [intro] , [verse] 在寂静的夜里.旋律悄然苏醒.像花瓣一样.轻轻绽放在风里.每一个音符.都是心跳的痕迹.带我走向未知的奇迹.无数色彩在空气中涌动.灵感在指尖跳跃成梦 , [chorus] SongBloom.让歌声绽放天空.点亮世界最温柔的心动.SongBloom.跨越时间的河流.用旋律把梦想托起.永不落空 , [inst] [inst] [inst] , [verse] 心里的秘密.化作和声交织.像春天的花海.彼此回应呼吸.世界在倾听.故事正在继续.把未来写进音乐里.无数色彩在空气中涌动.灵感在指尖跳跃成梦 , [chorus] SongBloom.让歌声绽放天空.点亮世界最温柔的心动.SongBloom.跨越时间的河流.用旋律把梦想托起.永不落空 ,  [inst] [inst] , [bridge] 即使黑夜再长.星光依然明亮.有歌声相伴.就有无限希望 , [chorus] SongBloom.让灵魂随风舞动.把每一颗心点亮成宇宙.SongBloom.让未来一起合奏.让世界听见我们的梦.永远相拥 , [outro] [outro] [outro] [outro]",
                        "example/test.wav",
                        1
                    ],
                ],
                inputs=[lyrics_input, prompt_audio_input, n_samples_input],
                label="预设示例",
                examples_per_page=3
            )
        
        # 绑定事件
        generate_btn.click(
            fn=generate_song,
            inputs=[lyrics_input, prompt_audio_input, n_samples_input],
            outputs=[output_audio, output_info]
        )
    
    return app


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 预加载模型
    try:
        initialize_model()
    except Exception as e:
        print(f"⚠️  模型初始化失败: {e}")
        print("将在第一次生成时加载模型")
    
    # 创建并启动界面
    print(f"\n启动 Gradio 界面...")
    print(f"访问地址: http://{HOST}:{PORT}\n")
    
    app = create_ui()
    app.launch(
        server_name=HOST,
        server_port=PORT,
        share=False
    )


if __name__ == "__main__":
    main()
