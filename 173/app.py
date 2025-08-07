#!/usr/bin/env python3
"""
SongGeneration WebUI - A Gradio interface for song generation
Based on generate.py but with flash_attn disabled for compatibility

配置说明:
- 修改下面的 CONFIG 部分来配置参数
- 直接运行: python webui_app.py
"""

# ===================== 配置区域 =====================
CONFIG = {
    # 模型检查点路径 (必须修改为正确路径)
    "CHECKPOINT_PATH": "ckpt/songgeneration_base",
    
    # 性能设置
    "USE_FLASH_ATTN": True,      # 是否启用 Flash Attention (推荐 True)
    "LOW_MEMORY_MODE": False,    # 是否启用低内存模式
    
    # 服务器配置
    "HOST": "0.0.0.0",           # 监听地址，0.0.0.0 表示所有网络接口
    "PORT": 7860,                # 端口号
    "SHARE": False,              # 是否创建公开链接 (Gradio share)
    
    # 其他设置
    "AUTO_OPEN_BROWSER": True,   # 是否自动打开浏览器
}
# =================================================

import os
import sys
import gc
import json
import time
import torch
import torchaudio
import numpy as np
import gradio as gr
from datetime import datetime
from omegaconf import OmegaConf
from codeclm.models import builders
from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from third_party.demucs.models.pretrained import get_model_from_yaml

def load_examples_from_jsonl(jsonl_path="sample/lyrics.jsonl"):
    """从 JSONL 文件加载示例数据"""
    examples = []
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        # 构建示例格式：[歌词, 描述, 音频路径, 自动提示类型]
                        lyric = data.get('gt_lyric', '')
                        description = data.get('descriptions', '')
                        audio_path = data.get('prompt_audio_path', '')
                        auto_prompt = data.get('auto_prompt_audio_type', 'None')
                        
                        # 示例名称基于 idx
                        idx = data.get('idx', f'sample_{len(examples)+1}')
                        
                        examples.append([
                            lyric,                          # 歌词
                            description,                    # 描述
                            audio_path if audio_path and os.path.exists(audio_path) else None,  # 音频路径
                            auto_prompt if auto_prompt != 'Auto' else 'Auto',  # 自动提示类型
                            "mixed",                        # 生成类型
                            1.5,                           # CFG 系数
                            0.9,                           # 温度
                            50,                            # Top-K
                            True,                          # 启用 Flash Attention
                            idx                            # 示例名称
                        ])
        except Exception as e:
            print(f"加载示例文件失败: {e}")
    
    # 如果没有找到示例文件或示例为空，添加默认示例
    if not examples:
        examples = [
            [
                "[intro-short]\n\n[verse]\n夜晚的街灯闪烁\n我漫步在熟悉的角落\n回忆像潮水般涌来\n你的笑容如此清晰\n\n[chorus]\n在心头无法抹去\n那些曾经的甜蜜\n如今只剩我独自回忆\n音乐的节奏在奏响\n\n[outro-short]",
                "female, dark, pop, sad, piano and drums",
                None,
                "Auto",
                "mixed",
                1.5,
                0.9,
                50,
                True,
                "默认示例"
            ]
        ]
    
    return examples

# 设置环境变量
os.environ['USER'] = 'root'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'third_party/hub')
os.environ['NCCL_HOME'] = '/usr/local/tccl'

# 添加到 Python 路径
current_dir = os.getcwd()
sys.path.extend([
    os.path.join(current_dir, 'codeclm/tokenizer'),
    current_dir,
    os.path.join(current_dir, 'codeclm/tokenizer/Flow1dVAE'),
])

# 禁用 cudnn 和注册 OmegaConf 解析器
torch.backends.cudnn.enabled = False
OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else 'default')
OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))

# 全局变量
auto_prompt_type = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']

class Separator:
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def load_audio(self, f):
        a, fs = torchaudio.load(f)
        if (fs != 48000):
            a = torchaudio.functional.resample(a, fs, 48000)
        if a.shape[-1] >= 48000*10:
            a = a[..., :48000*10]
        return a[:, 0:48000*10]
    
    def run(self, audio_path, output_dir='tmp', ext=".flac"):
        os.makedirs(output_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []

        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 4:
            drums_path, bass_path, other_path, vocal_path = output_paths
        else:
            drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
            for path in [drums_path, bass_path, other_path]:
                os.remove(path)
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        bgm_audio = full_audio - vocal_audio
        return full_audio, vocal_audio, bgm_audio


class SongGenerator:
    def __init__(self, ckpt_path, use_flash_attn=True):
        self.ckpt_path = ckpt_path
        self.cfg_path = os.path.join(ckpt_path, 'config.yaml')
        self.pt_path = os.path.join(ckpt_path, 'model.pt')
        
        # 加载配置
        self.cfg = OmegaConf.load(self.cfg_path)
        self.cfg.mode = 'inference'
        
        # 配置 flash_attn - 现在启用它来提升性能
        if hasattr(self.cfg, 'lm'):
            if use_flash_attn:
                # 优先使用 flash_attn_2，如果不可用则回退到原版
                if hasattr(self.cfg.lm, 'use_flash_attn_2'):
                    self.cfg.lm.use_flash_attn_2 = True
                elif hasattr(self.cfg.lm, 'use_flash_attn'):
                    self.cfg.lm.use_flash_attn = True
                print("Flash Attention enabled for better performance")
            else:
                if hasattr(self.cfg.lm, 'use_flash_attn_2'):
                    self.cfg.lm.use_flash_attn_2 = False
                if hasattr(self.cfg.lm, 'use_flash_attn'):
                    self.cfg.lm.use_flash_attn = False
                print("Flash Attention disabled for compatibility")
        
        self.max_duration = self.cfg.max_dur
        self.separator = Separator()
        
        # 加载自动提示
        try:
            self.auto_prompt = torch.load('ckpt/prompt.pt')
            self.merge_prompt = [item for sublist in self.auto_prompt.values() for item in sublist]
        except:
            print("Warning: Could not load auto prompt file")
            self.auto_prompt = {}
            self.merge_prompt = []

    def generate_song(self, lyric, description=None, prompt_audio_path=None, auto_prompt_type_select=None, 
                     generate_type="mixed", cfg_coef=1.5, temperature=0.9, top_k=50, progress=gr.Progress()):
        """生成歌曲的主要函数"""
        try:
            progress(0.1, "准备生成参数...")
            
            # 创建临时输出目录
            save_dir = f"tmp_output_{int(time.time())}"
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(f"{save_dir}/audios", exist_ok=True)
            
            # 准备生成项目
            item = {
                'idx': f'webui_{int(time.time())}',
                'gt_lyric': lyric.replace("  ", " "),
                'descriptions': description if description else None
            }
            
            progress(0.2, "处理音频提示...")
            
            # 处理音频提示
            if prompt_audio_path and os.path.exists(prompt_audio_path):
                with torch.no_grad():
                    pmt_wav, vocal_wav, bgm_wav = self.separator.run(prompt_audio_path)
                    
                # 编码音频
                audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
                audio_tokenizer = audio_tokenizer.eval().cuda()
                
                if pmt_wav.dim() == 2:
                    pmt_wav = pmt_wav[None]
                if vocal_wav.dim() == 2:
                    vocal_wav = vocal_wav[None]
                if bgm_wav.dim() == 2:
                    bgm_wav = bgm_wav[None]
                    
                with torch.no_grad():
                    pmt_wav, _ = audio_tokenizer.encode(pmt_wav.cuda())
                    
                del audio_tokenizer
                torch.cuda.empty_cache()
                
                # 分离音频编码
                if "audio_tokenizer_checkpoint_sep" in self.cfg.keys():
                    seperate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
                    seperate_tokenizer = seperate_tokenizer.eval().cuda()
                    with torch.no_grad():
                        vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_wav.cuda(), bgm_wav.cuda())
                    del seperate_tokenizer
                    torch.cuda.empty_cache()
                
                melody_is_wav = False
                
            elif auto_prompt_type_select and auto_prompt_type_select != "None" and self.auto_prompt:
                if auto_prompt_type_select == "Auto": 
                    prompt_token = self.merge_prompt[np.random.randint(0, len(self.merge_prompt))]
                else:
                    prompt_token = self.auto_prompt[auto_prompt_type_select][np.random.randint(0, len(self.auto_prompt[auto_prompt_type_select]))]
                pmt_wav = prompt_token[:,[0],:]
                vocal_wav = prompt_token[:,[1],:]
                bgm_wav = prompt_token[:,[2],:]
                melody_is_wav = False
            else:
                pmt_wav = None
                vocal_wav = None
                bgm_wav = None
                melody_is_wav = True

            progress(0.4, "加载语言模型...")
            
            # 加载语言模型
            audiolm = builders.get_lm_model(self.cfg)
            checkpoint = torch.load(self.pt_path, map_location='cpu')
            audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
            audiolm.load_state_dict(audiolm_state_dict, strict=False)
            audiolm = audiolm.eval().cuda().to(torch.float16)

            model = CodecLM(
                name="webui_tmp",
                lm=audiolm,
                audiotokenizer=None,
                max_duration=self.max_duration,
                seperate_tokenizer=None,
            )
            
            # 设置生成参数
            model.set_generation_params(
                duration=self.max_duration, 
                extend_stride=5, 
                temperature=temperature, 
                cfg_coef=cfg_coef,
                top_k=top_k, 
                top_p=0.0, 
                record_tokens=True, 
                record_window=50
            )

            progress(0.6, "生成音乐...")
            
            # 生成
            generate_inp = {
                'lyrics': [lyric.replace("  ", " ")],
                'descriptions': [description],
                'melody_wavs': pmt_wav,
                'vocal_wavs': vocal_wav,
                'bgm_wavs': bgm_wav,
                'melody_is_wav': melody_is_wav,
            }
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    tokens = model.generate(**generate_inp, return_tokens=True)

            # 清理内存
            del model
            audiolm = audiolm.cpu()
            del audiolm
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()

            progress(0.8, "生成音频...")
            
            # 生成音频
            seperate_tokenizer = builders.get_audio_tokenizer_model_cpu(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            device = "cuda:0"
            seperate_tokenizer.model.device = device
            seperate_tokenizer.model.vae = seperate_tokenizer.model.vae.to(device)
            seperate_tokenizer.model.model.device = torch.device(device)
            seperate_tokenizer.model.model = seperate_tokenizer.model.model.to(device)
            seperate_tokenizer = seperate_tokenizer.eval()

            model = CodecLM(
                name="webui_tmp",
                lm=None,
                audiotokenizer=None,
                max_duration=self.max_duration,
                seperate_tokenizer=seperate_tokenizer,
            )

            with torch.no_grad():
                if generate_type == 'separate':
                    wav_vocal = model.generate_audio(tokens, chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, chunked=True, gen_type='bgm')
                    wav_mixed = model.generate_audio(tokens, chunked=True, gen_type='mixed')
                    
                    # 保存文件
                    output_files = {}
                    output_files['vocal'] = f"{save_dir}/audios/vocal.flac"
                    output_files['bgm'] = f"{save_dir}/audios/bgm.flac"
                    output_files['mixed'] = f"{save_dir}/audios/mixed.flac"
                    
                    torchaudio.save(output_files['vocal'], wav_vocal[0].cpu().float(), self.cfg.sample_rate)
                    torchaudio.save(output_files['bgm'], wav_bgm[0].cpu().float(), self.cfg.sample_rate)
                    torchaudio.save(output_files['mixed'], wav_mixed[0].cpu().float(), self.cfg.sample_rate)
                    
                    # 返回混合音频
                    result_audio = (self.cfg.sample_rate, wav_mixed[0].cpu().float().numpy().T)
                    
                else:
                    wav_result = model.generate_audio(tokens, chunked=True, gen_type=generate_type)
                    output_file = f"{save_dir}/audios/result.flac"
                    torchaudio.save(output_file, wav_result[0].cpu().float(), self.cfg.sample_rate)
                    result_audio = (self.cfg.sample_rate, wav_result[0].cpu().float().numpy().T)

            torch.cuda.empty_cache()
            
            progress(1.0, "生成完成!")
            
            # 创建结果信息
            result_info = {
                "lyric": lyric,
                "description": description,
                "prompt_audio": prompt_audio_path,
                "auto_prompt_type": auto_prompt_type_select,
                "generate_type": generate_type,
                "cfg_coef": cfg_coef,
                "temperature": temperature,
                "top_k": top_k,
                "timestamp": datetime.now().isoformat(),
                "sample_rate": self.cfg.sample_rate
            }
            
            return result_audio, json.dumps(result_info, indent=2, ensure_ascii=False)
            
        except Exception as e:
            error_msg = f"生成失败: {str(e)}"
            print(error_msg)
            return None, json.dumps({"error": error_msg}, ensure_ascii=False)


# 全局生成器实例
generator = None

def initialize_generator(ckpt_path, use_flash_attn=True):
    """初始化生成器"""
    global generator
    if generator is None:
        generator = SongGenerator(ckpt_path, use_flash_attn)
    return generator

def generate_song_wrapper(lyric, description, prompt_audio, auto_prompt_type_select, 
                         generate_type, cfg_coef, temperature, top_k, use_flash_attn, progress=gr.Progress()):
    """Gradio 包装函数"""
    global generator
    
    if generator is None:
        return None, json.dumps({"error": "生成器未初始化"}, ensure_ascii=False)
    
    try:
        # 如果 flash_attn 设置与当前不同，重新初始化生成器
        current_flash_attn = getattr(generator.cfg.lm, 'use_flash_attn_2', getattr(generator.cfg.lm, 'use_flash_attn', False))
        if current_flash_attn != use_flash_attn:
            generator = None
            initialize_generator(CONFIG["CHECKPOINT_PATH"], use_flash_attn)
            print(f"Flash Attention 设置已更新为: {'启用' if use_flash_attn else '禁用'}")
        
        # 创建输出目录（使用时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/webui_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成歌曲
        result_audio, info = generator.generate_song(
            lyric=lyric,
            description=description,
            prompt_audio_path=prompt_audio,
            auto_prompt_type_select=auto_prompt_type_select,
            generate_type=generate_type,
            cfg_coef=cfg_coef,
            temperature=temperature,
            top_k=top_k,
            progress=progress
        )
        
        # 如果生成成功，保存到输出目录
        if result_audio is not None:
            # 保存音频文件
            audio_filename = f"generated_{timestamp}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            
            # 使用 torchaudio 保存
            sample_rate, audio_data = result_audio
            audio_tensor = torch.from_numpy(audio_data.T).float()  # 转换为 (channels, samples) 格式
            torchaudio.save(audio_path, audio_tensor, sample_rate)
            
            # 保存生成信息
            info_filename = f"info_{timestamp}.json"
            info_path = os.path.join(output_dir, info_filename)
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(info)
            
            # 更新信息，包含文件路径
            info_dict = json.loads(info)
            info_dict["output_directory"] = output_dir
            info_dict["audio_file"] = audio_path
            info_dict["info_file"] = info_path
            info_dict["flash_attn_enabled"] = use_flash_attn
            updated_info = json.dumps(info_dict, indent=2, ensure_ascii=False)
            
            return result_audio, updated_info
        
        return result_audio, info
        
    except Exception as e:
        error_msg = f"生成过程中发生错误: {str(e)}"
        print(error_msg)
        return None, json.dumps({"error": error_msg}, ensure_ascii=False)

# 示例歌词
EXAMPLE_LYRICS = """[intro-short]

[verse]
夜晚的街灯闪烁
我漫步在熟悉的角落
回忆像潮水般涌来
你的笑容如此清晰

[chorus]
在心头无法抹去
那些曾经的甜蜜
如今只剩我独自回忆
音乐的节奏在奏响

[verse]
手机屏幕亮起
是你发来的消息
简单的几个字
却让我泪流满面

[chorus]
回忆的温度还在
你却已不在
我的心被爱填满
却又被思念刺痛

[outro-short]"""

def create_gradio_interface():
    """创建 Gradio 界面"""
    # 加载示例数据
    examples_data = load_examples_from_jsonl()
    
    with gr.Blocks(title="SongGeneration WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎵 SongGeneration WebUI")
        gr.Markdown("基于腾讯 AI Lab 的 SongGeneration 模型的网页界面，支持 Flash Attention 加速")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 歌词输入
                lyric_input = gr.Textbox(
                    label="歌词 (Lyrics)",
                    lines=10,
                    max_lines=20,
                    value=EXAMPLE_LYRICS,
                    placeholder="请输入歌词...",
                    info="每个段落代表一个结构段，以结构标签开始"
                )
                
                # 提示设置
                with gr.Tabs():
                    with gr.Tab("文本描述"):
                        description_input = gr.Textbox(
                            label="歌曲描述 (可选)",
                            placeholder="例如: female, dark, pop, sad, piano and drums, the bpm is 125",
                            info="描述歌曲的性别、音色、风格、情绪、乐器和 BPM"
                        )
                    
                    with gr.Tab("音频提示"):
                        prompt_audio_input = gr.Audio(
                            label="提示音频 (可选)",
                            type="filepath"
                        )
                    
                    with gr.Tab("自动提示"):
                        auto_prompt_select = gr.Dropdown(
                            choices=["None"] + auto_prompt_type,
                            value="Auto",
                            label="自动提示类型",
                            info="选择音乐风格进行自动提示"
                        )
                
                # 生成设置
                with gr.Accordion("生成设置", open=False):
                    generate_type_input = gr.Radio(
                        choices=["mixed", "vocal", "bgm", "separate"],
                        value="mixed",
                        label="生成类型",
                        info="mixed: 完整歌曲, vocal: 人声, bgm: 背景音乐, separate: 分离生成"
                    )
                    
                    with gr.Row():
                        cfg_coef_input = gr.Slider(
                            minimum=0.1,
                            maximum=3.0,
                            step=0.1,
                            value=1.5,
                            label="CFG 系数"
                        )
                        
                        temperature_input = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            value=0.9,
                            label="温度"
                        )
                    
                    with gr.Row():
                        top_k_input = gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50,
                            label="Top-K"
                        )
                        
                        use_flash_attn_input = gr.Checkbox(
                            value=CONFIG.get("USE_FLASH_ATTN", True),
                            label="启用 Flash Attention",
                            info="提升生成速度，需要兼容的 GPU"
                        )
                
                # 生成按钮
                generate_btn = gr.Button("🎵 生成歌曲", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # 输出
                output_audio = gr.Audio(
                    label="生成的音频",
                    type="numpy"
                )
                
                output_info = gr.JSON(
                    label="生成信息"
                )
        
        # 示例区域
        with gr.Row():
            gr.Markdown("## 📚 示例库")
        
        with gr.Row():
            # 创建示例组件
            example_inputs = [
                lyric_input,
                description_input, 
                prompt_audio_input,
                auto_prompt_select,
                generate_type_input,
                cfg_coef_input,
                temperature_input,
                top_k_input,
                use_flash_attn_input
            ]
            
            # 只使用前9个元素作为示例输入（去掉示例名称）
            examples_for_gradio = [example[:9] for example in examples_data]
            
            gr.Examples(
                examples=examples_for_gradio,
                inputs=example_inputs,
                label="点击下方示例快速填充参数:",
                examples_per_page=5
            )
        
        # 绑定事件
        generate_btn.click(
            fn=generate_song_wrapper,
            inputs=[
                lyric_input,
                description_input,
                prompt_audio_input,
                auto_prompt_select,
                generate_type_input,
                cfg_coef_input,
                temperature_input,
                top_k_input,
                use_flash_attn_input
            ],
            outputs=[output_audio, output_info]
        )
        
        # 添加使用说明
        with gr.Accordion("� 使用说明", open=False):
            gr.Markdown("""
            ### 🎯 基本使用
            1. **选择示例**: 点击下方示例库中的任一示例快速填充参数
            2. **修改参数**: 根据需要调整歌词、描述和生成设置
            3. **生成歌曲**: 点击"生成歌曲"按钮开始创作
            4. **预览播放**: 生成完成后可在右侧直接播放预览
            
            ### 📝 歌词格式
            - 使用结构标签：`[intro-short]`, `[intro-medium]`, `[intro-long]`
            - 主歌：`[verse]`，副歌：`[chorus]`，桥段：`[bridge]`
            - 间奏：`[inst-short]`, `[inst-medium]`, `[inst-long]`
            - 结尾：`[outro-short]`, `[outro-medium]`, `[outro-long]`
            - 静音：`[silence]`
            
            ### 🎛️ 提示方式
            - **文本描述**: 描述性别、风格、情绪、乐器等
            - **音频提示**: 上传参考音频文件
            - **自动提示**: 选择预设的音乐风格
            
            ### 🎵 生成类型
            - **mixed**: 完整歌曲（推荐）
            - **vocal**: 只生成人声部分  
            - **bgm**: 只生成背景音乐
            - **separate**: 同时生成人声、背景音乐和混合版本
            
            ### ⚡ 性能优化
            - **Flash Attention**: 现已支持并默认启用，显著提升生成速度
            - **低内存模式**: 适用于显存较小的设备
            - **CUDA 加速**: 自动检测并使用 GPU 加速
            - **动态内存管理**: 自动清理显存，避免内存溢出
            
            ### 🔧 高级设置
            - **CFG 系数**: 控制生成与提示的符合度 (1.0-3.0)
            - **温度**: 控制生成的随机性 (0.1-2.0，越低越保守)
            - **Top-K**: 限制候选词汇数量 (1-100)
            - **Flash Attention**: 可动态开关，重启时生效
            """)
    
    return demo

def main():
    # 从配置中读取参数
    CHECKPOINT_PATH = CONFIG["CHECKPOINT_PATH"]
    USE_FLASH_ATTN = CONFIG["USE_FLASH_ATTN"]
    LOW_MEMORY_MODE = CONFIG["LOW_MEMORY_MODE"]
    HOST = CONFIG["HOST"]
    PORT = CONFIG["PORT"]
    SHARE = CONFIG["SHARE"]
    AUTO_OPEN_BROWSER = CONFIG["AUTO_OPEN_BROWSER"]
    
    # 检查检查点路径
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误: 检查点路径不存在: {CHECKPOINT_PATH}")
        print("请修改 webui_app.py 中的 CONFIG['CHECKPOINT_PATH'] 变量")
        print("例如: CONFIG['CHECKPOINT_PATH'] = 'ckpt/songgeneration_base'")
        sys.exit(1)
    
    if not os.path.exists(os.path.join(CHECKPOINT_PATH, 'config.yaml')):
        print(f"错误: 在 {CHECKPOINT_PATH} 中找不到 config.yaml")
        sys.exit(1)
    
    if not os.path.exists(os.path.join(CHECKPOINT_PATH, 'model.pt')):
        print(f"错误: 在 {CHECKPOINT_PATH} 中找不到 model.pt")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    print("=== SongGeneration WebUI ===")
    print(f"检查点路径: {CHECKPOINT_PATH}")
    print(f"Flash Attention: {'启用' if USE_FLASH_ATTN else '禁用'}")
    print(f"低内存模式: {'启用' if LOW_MEMORY_MODE else '禁用'}")
    print(f"启动地址: http://{HOST}:{PORT}")
    print(f"输出目录: outputs/")
    if SHARE:
        print("公开链接: 启用")
    print("============================")
    
    # 设置随机种子
    np.random.seed(int(time.time()))
    
    # 初始化生成器
    try:
        print("正在初始化生成器...")
        initialize_generator(CHECKPOINT_PATH, USE_FLASH_ATTN)
        print("生成器初始化成功!")
    except Exception as e:
        print(f"生成器初始化失败: {e}")
        print("请检查:")
        print("1. 检查点路径是否正确")
        print("2. 是否安装了所有依赖")
        print("3. 是否有足够的 GPU 内存")
        print("4. Flash Attention 是否正确安装")
        sys.exit(1)
    
    # 创建并启动界面
    demo = create_gradio_interface()
    
    demo.launch(
        server_name=HOST,
        server_port=PORT,
        share=SHARE,
        inbrowser=AUTO_OPEN_BROWSER
    )

if __name__ == "__main__":
    main()
