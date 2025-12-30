"""
Chatterbox TTS 统一应用
支持多语言TTS和Turbo TTS两种模式
"""
import os
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
import gradio as gr

# 设备检测
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 运行设备: {DEVICE}")

# 本地模型路径
CKPT_DIR = Path(__file__).parent / "checkpoints"
MTL_CKPT_DIR = CKPT_DIR / "chatterbox"
TURBO_CKPT_DIR = CKPT_DIR / "chatterbox-turbo"

# 示例音频目录
EXAMPLES_DIR = Path(__file__).parent / "examples"

# ==================== 示例音频配置 ====================
# 多语言示例配置
LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑。 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}

# Turbo 示例配置
TURBO_EXAMPLE = {
    "audio": "https://storage.googleapis.com/chatterbox-demo-samples/turbo/2.wav",
    "text": "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?"
}

# Turbo 事件标签
EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

# 支持的语言
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

# ==================== 示例音频下载 ====================
def download_example_audio():
    """下载示例音频文件到 examples 目录"""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 下载多语言示例
    mtl_dir = EXAMPLES_DIR / "multilingual"
    mtl_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 正在下载多语言示例音频...")
    for lang_code, config in LANGUAGE_CONFIG.items():
        url = config["audio"]
        # 从URL提取文件扩展名
        ext = url.split(".")[-1]
        local_path = mtl_dir / f"{lang_code}_example.{ext}"
        
        if not local_path.exists():
            try:
                print(f"  下载 {lang_code} 示例: {url}")
                urllib.request.urlretrieve(url, local_path)
                # 更新配置为本地路径
                LANGUAGE_CONFIG[lang_code]["local_audio"] = str(local_path)
            except Exception as e:
                print(f"  ⚠️ 下载 {lang_code} 示例失败: {e}")
                LANGUAGE_CONFIG[lang_code]["local_audio"] = url  # 回退到URL
        else:
            print(f"  ✓ {lang_code} 示例已存在")
            LANGUAGE_CONFIG[lang_code]["local_audio"] = str(local_path)
    
    # 下载 Turbo 示例
    turbo_dir = EXAMPLES_DIR / "turbo"
    turbo_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 正在下载 Turbo 示例音频...")
    turbo_url = TURBO_EXAMPLE["audio"]
    turbo_local_path = turbo_dir / "example.wav"
    
    if not turbo_local_path.exists():
        try:
            print(f"  下载 Turbo 示例: {turbo_url}")
            urllib.request.urlretrieve(turbo_url, turbo_local_path)
            TURBO_EXAMPLE["local_audio"] = str(turbo_local_path)
        except Exception as e:
            print(f"  ⚠️ 下载 Turbo 示例失败: {e}")
            TURBO_EXAMPLE["local_audio"] = turbo_url
    else:
        print(f"  ✓ Turbo 示例已存在")
        TURBO_EXAMPLE["local_audio"] = str(turbo_local_path)
    
    print("✅ 示例音频下载完成!")

# 启动时下载示例音频
download_example_audio()

# ==================== 模型加载 ====================
MTL_MODEL = None
TURBO_MODEL = None

def load_mtl_model():
    """加载多语言 TTS 模型"""
    global MTL_MODEL
    if MTL_MODEL is None:
        print("🔄 正在加载多语言 TTS 模型...")
        from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
        MTL_MODEL = ChatterboxMultilingualTTS.from_local(MTL_CKPT_DIR, DEVICE)
        print("✅ 多语言 TTS 模型加载完成!")
    return MTL_MODEL

def load_turbo_model():
    """加载 Turbo TTS 模型"""
    global TURBO_MODEL
    if TURBO_MODEL is None:
        print("🔄 正在加载 Turbo TTS 模型...")
        from src.chatterbox.tts_turbo import ChatterboxTurboTTS
        TURBO_MODEL = ChatterboxTurboTTS.from_local(TURBO_CKPT_DIR, DEVICE)
        print("✅ Turbo TTS 模型加载完成!")
    return TURBO_MODEL

# ==================== 工具函数 ====================
def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_mtl_audio_for_lang(lang: str) -> str | None:
    """获取指定语言的示例音频路径"""
    config = LANGUAGE_CONFIG.get(lang, {})
    return config.get("local_audio", config.get("audio"))

def get_mtl_text_for_lang(lang: str) -> str:
    """获取指定语言的示例文本"""
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")

def get_supported_languages_display() -> str:
    """生成支持语言的格式化显示"""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")
    
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    
    return f"""
### 🌍 支持的语言 ({len(SUPPORTED_LANGUAGES)} 种)
{line1}
{line2}
"""

# ==================== 多语言 TTS 生成 ====================
def generate_mtl_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    """
    使用多语言模型生成语音
    """
    model = load_mtl_model()
    
    if seed_num_input != 0:
        set_seed(int(seed_num_input))
    
    print(f"🎤 正在生成音频，文本: '{text_input[:50]}...'")
    
    # 处理音频提示
    chosen_prompt = audio_prompt_path_input or get_mtl_audio_for_lang(language_id)
    
    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }
    
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"  使用参考音频: {chosen_prompt}")
    else:
        print("  使用默认声音")
    
    wav = model.generate(
        text_input[:300],
        language_id=language_id,
        **generate_kwargs
    )
    
    print("✅ 音频生成完成!")
    return (model.sr, wav.squeeze(0).numpy())

def on_mtl_language_change(lang, current_ref, current_text):
    """语言变更时更新参考音频和文本"""
    return get_mtl_audio_for_lang(lang), get_mtl_text_for_lang(lang)

# ==================== Turbo TTS 生成 ====================
def generate_turbo_audio(
    text: str,
    audio_prompt_path: str,
    temperature: float,
    seed_num: int,
    min_p: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    norm_loudness: bool
) -> tuple[int, np.ndarray]:
    """
    使用 Turbo 模型生成语音
    """
    model = load_turbo_model()
    
    if seed_num != 0:
        set_seed(int(seed_num))
    
    print(f"⚡ 正在生成 Turbo 音频，文本: '{text[:50]}...'")
    
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
        norm_loudness=norm_loudness,
    )
    
    print("✅ Turbo 音频生成完成!")
    return (model.sr, wav.squeeze(0).cpu().numpy())

# ==================== 自定义 CSS ====================
CUSTOM_CSS = """
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
.audio-note {
    font-size: 0.9em;
    color: #666;
}
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#turbo_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# ==================== 构建 Gradio 界面 ====================
with gr.Blocks(title="Chatterbox TTS", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        """
        # 🎙️ Chatterbox TTS 语音合成
        高质量文本转语音系统，支持多语言合成和快速 Turbo 模式。
        """
    )
    
    with gr.Tabs():
        # ==================== 多语言 TTS 标签页 ====================
        with gr.TabItem("🌍 多语言 TTS"):
            gr.Markdown(
                """
                ### 多语言语音合成
                支持 23 种语言的高质量语音合成，可使用参考音频进行声音克隆。
                """
            )
            gr.Markdown(get_supported_languages_display())
            
            with gr.Row():
                with gr.Column():
                    initial_lang = "zh"
                    mtl_text = gr.Textbox(
                        value=get_mtl_text_for_lang(initial_lang),
                        label="合成文本 (最多 300 字符)",
                        max_lines=5
                    )
                    
                    mtl_language = gr.Dropdown(
                        choices=list(SUPPORTED_LANGUAGES.keys()),
                        value=initial_lang,
                        label="语言",
                        info="选择语音合成的语言"
                    )
                    
                    mtl_ref_wav = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="参考音频 (可选)",
                        value=get_mtl_audio_for_lang(initial_lang)
                    )
                    
                    gr.Markdown(
                        "💡 **提示**: 确保参考音频的语言与选择的语言标签匹配。否则，语言转换输出可能会继承参考音频语言的口音。要减轻这种情况，可以将 CFG 权重设置为 0。",
                        elem_classes=["audio-note"]
                    )
                    
                    mtl_exaggeration = gr.Slider(
                        0.25, 2, step=0.05,
                        label="夸张度 (0.5 为中性，极端值可能不稳定)",
                        value=0.5
                    )
                    
                    mtl_cfg_weight = gr.Slider(
                        0.2, 1, step=0.05,
                        label="CFG/节奏",
                        value=0.5
                    )
                    
                    with gr.Accordion("更多选项", open=False):
                        mtl_seed = gr.Number(value=0, label="随机种子 (0 表示随机)")
                        mtl_temp = gr.Slider(0.05, 5, step=0.05, label="温度", value=0.8)
                    
                    mtl_run_btn = gr.Button("🎤 生成语音", variant="primary")
                
                with gr.Column():
                    mtl_audio_output = gr.Audio(label="输出音频")
            
            # 语言变更事件
            mtl_language.change(
                fn=on_mtl_language_change,
                inputs=[mtl_language, mtl_ref_wav, mtl_text],
                outputs=[mtl_ref_wav, mtl_text],
                show_progress=False
            )
            
            # 生成按钮事件
            mtl_run_btn.click(
                fn=generate_mtl_audio,
                inputs=[
                    mtl_text,
                    mtl_language,
                    mtl_ref_wav,
                    mtl_exaggeration,
                    mtl_temp,
                    mtl_seed,
                    mtl_cfg_weight,
                ],
                outputs=[mtl_audio_output],
            )
        
        # ==================== Turbo TTS 标签页 ====================
        with gr.TabItem("⚡ Turbo TTS"):
            gr.Markdown(
                """
                ### 快速语音合成 (Turbo)
                超快速英语语音合成，支持情感标签插入。
                """
            )
            
            with gr.Row():
                with gr.Column():
                    turbo_text = gr.Textbox(
                        value=TURBO_EXAMPLE["text"],
                        label="合成文本 (最多 300 字符)",
                        max_lines=5,
                        elem_id="turbo_textbox"
                    )
                    
                    gr.Markdown("**插入情感/事件标签:**")
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, turbo_text],
                                outputs=turbo_text,
                                js=INSERT_TAG_JS
                            )
                    
                    turbo_ref_wav = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="参考音频",
                        value=TURBO_EXAMPLE.get("local_audio", TURBO_EXAMPLE["audio"])
                    )
                    
                    turbo_run_btn = gr.Button("⚡ 快速生成", variant="primary")
                
                with gr.Column():
                    turbo_audio_output = gr.Audio(label="输出音频")
                    
                    with gr.Accordion("高级选项", open=False):
                        turbo_seed = gr.Number(value=0, label="随机种子 (0 表示随机)")
                        turbo_temp = gr.Slider(0.05, 2.0, step=0.05, label="温度", value=0.8)
                        turbo_top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                        turbo_top_k = gr.Slider(0, 1000, step=10, label="Top K", value=1000)
                        turbo_rep_penalty = gr.Slider(1.00, 2.00, step=0.05, label="重复惩罚", value=1.2)
                        turbo_min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (设为 0 禁用)", value=0.00)
                        turbo_norm_loudness = gr.Checkbox(value=True, label="归一化响度 (-27 LUFS)")
            
            # Turbo 生成按钮事件
            turbo_run_btn.click(
                fn=generate_turbo_audio,
                inputs=[
                    turbo_text,
                    turbo_ref_wav,
                    turbo_temp,
                    turbo_seed,
                    turbo_min_p,
                    turbo_top_p,
                    turbo_top_k,
                    turbo_rep_penalty,
                    turbo_norm_loudness,
                ],
                outputs=turbo_audio_output,
            )
    
    gr.Markdown(
        """
        ---
        💡 **使用说明**:
        - **多语言 TTS**: 支持 23 种语言，可上传参考音频进行声音克隆
        - **Turbo TTS**: 专为英语优化的快速合成模式，支持情感标签
        - 参考音频越清晰，合成效果越好
        - 建议参考音频时长为 5-15 秒
        """
    )

if __name__ == "__main__":
    # 预加载模型（可选，可以减少首次生成的等待时间）
    print("\n" + "="*50)
    print("🚀 启动 Chatterbox TTS 服务...")
    print("="*50 + "\n")
    
    # 可以选择预加载模型，但会增加启动时间
    # load_mtl_model()
    # load_turbo_model()
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
