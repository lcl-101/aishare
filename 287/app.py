import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型路径
MODEL_PATH = "checkpoints/HY-MT1.5-7B"

# 支持的语言列表
SUPPORTED_LANGUAGES = {
    "中文": "Chinese",
    "英语": "English",
    "法语": "French",
    "葡萄牙语": "Portuguese",
    "西班牙语": "Spanish",
    "日语": "Japanese",
    "土耳其语": "Turkish",
    "俄语": "Russian",
    "阿拉伯语": "Arabic",
    "韩语": "Korean",
    "泰语": "Thai",
    "意大利语": "Italian",
    "德语": "German",
    "越南语": "Vietnamese",
    "马来语": "Malay",
    "印尼语": "Indonesian",
    "菲律宾语": "Filipino",
    "印地语": "Hindi",
    "繁体中文": "Traditional Chinese",
    "波兰语": "Polish",
    "捷克语": "Czech",
    "荷兰语": "Dutch",
    "高棉语": "Khmer",
    "缅甸语": "Burmese",
    "波斯语": "Persian",
    "古吉拉特语": "Gujarati",
    "乌尔都语": "Urdu",
    "泰卢固语": "Telugu",
    "马拉地语": "Marathi",
    "希伯来语": "Hebrew",
    "孟加拉语": "Bengali",
    "泰米尔语": "Tamil",
    "乌克兰语": "Ukrainian",
    "藏语": "Tibetan",
    "哈萨克语": "Kazakh",
    "蒙古语": "Mongolian",
    "维吾尔语": "Uyghur",
    "粤语": "Cantonese",
}

# 加载模型和分词器
print("正在加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("模型加载完成！")


def translate(text: str, target_language: str) -> str:
    """
    翻译函数
    """
    if not text.strip():
        return "请输入要翻译的文本"
    
    target_lang_en = SUPPORTED_LANGUAGES.get(target_language, "Chinese")
    
    # 构建提示词
    prompt = f"Translate the following segment into {target_lang_en}, without additional explanation.\n\n{text}"
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    # 应用聊天模板
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # 推理参数
    generation_config = {
        "max_new_tokens": 2048,
        "top_k": 20,
        "top_p": 0.6,
        "repetition_penalty": 1.05,
        "temperature": 0.7,
        "do_sample": True,
    }
    
    # 生成翻译
    with torch.no_grad():
        outputs = model.generate(
            tokenized_chat.to(model.device),
            **generation_config
        )
    
    # 解码输出
    input_length = tokenized_chat.shape[1]
    generated_tokens = outputs[0][input_length:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return output_text.strip()


# 创建 Gradio 界面
with gr.Blocks(
    title="HY-MT 翻译系统",
    theme=gr.themes.Soft(),
    css="""
    .youtube-banner {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .youtube-banner a {
        color: #a0d8f1;
        text-decoration: underline;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        color: #666;
    }
    """
) as demo:
    
    # YouTube 频道宣传横幅
    gr.HTML("""
    <div class="youtube-banner">
        <h3>🎬 AI 技术分享频道</h3>
        <p>🔥 更多最新 AI 技术分享、模型部署教程、实战项目演示，尽在 YouTube 频道！</p>
        <p>👉 <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">https://www.youtube.com/@rongyikanshijie-ai</a></p>
        <p>📢 欢迎订阅，开启您的 AI 学习之旅！</p>
    </div>
    """)
    
    gr.Markdown("""
    # 🌐 HY-MT 多语言翻译系统
    
    基于腾讯混元 HY-MT1.5-7B 模型的智能翻译工具，支持 38 种语言互译。
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="📝 输入文本",
                placeholder="请输入要翻译的文本...",
                lines=8,
                max_lines=20,
            )
            
            target_lang = gr.Dropdown(
                choices=list(SUPPORTED_LANGUAGES.keys()),
                value="中文",
                label="🎯 目标语言",
                info="选择您想要翻译成的语言"
            )
            
            translate_btn = gr.Button("🚀 开始翻译", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="📄 翻译结果",
                placeholder="翻译结果将显示在这里...",
                lines=8,
                max_lines=20,
                interactive=False,
            )
    
    # 示例
    gr.Examples(
        examples=[
            ["It's on the house.", "中文"],
            ["今天天气真好，我们去公园散步吧。", "英语"],
            ["人工智能正在改变我们的生活方式。", "日语"],
            ["The quick brown fox jumps over the lazy dog.", "法语"],
            ["机器翻译技术已经取得了巨大的进步。", "韩语"],
        ],
        inputs=[input_text, target_lang],
        outputs=output_text,
        fn=translate,
        cache_examples=False,
    )
    
    # 绑定翻译按钮事件
    translate_btn.click(
        fn=translate,
        inputs=[input_text, target_lang],
        outputs=output_text,
    )
    
    # 支持回车键翻译
    input_text.submit(
        fn=translate,
        inputs=[input_text, target_lang],
        outputs=output_text,
    )
    
    gr.Markdown("""
    ---
    ### 📋 支持的语言列表
    
    中文、英语、法语、葡萄牙语、西班牙语、日语、土耳其语、俄语、阿拉伯语、韩语、
    泰语、意大利语、德语、越南语、马来语、印尼语、菲律宾语、印地语、繁体中文、波兰语、
    捷克语、荷兰语、高棉语、缅甸语、波斯语、古吉拉特语、乌尔都语、泰卢固语、马拉地语、
    希伯来语、孟加拉语、泰米尔语、乌克兰语、藏语、哈萨克语、蒙古语、维吾尔语、粤语
    
    ---
    """)
    
    # 页脚
    gr.HTML("""
    <div class="footer">
        <p>🔗 更多精彩内容请访问: <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">AI 技术分享频道</a></p>
        <p>💡 基于腾讯混元 HY-MT1.5-7B 模型构建</p>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
