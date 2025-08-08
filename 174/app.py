import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import gradio as gr
import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from fish_speech.i18n import i18n
from tools.webui.inference import get_inference_wrapper

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"

HEADER_MD = f"""# OpenAudio S1 - 语音合成

{i18n("A text-to-speech model based on VQ-GAN and Llama developed by [Fish Audio](https://fish.audio).")}  

{i18n("You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/fish-speech-1.5).")}  

{i18n("Related code and weights are released under CC BY-NC-SA 4.0 License.")}  

{i18n("We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.")}  

## 情感与控制标记

OpenAudio S1 支持多种情感、语调和特殊标记来增强语音合成效果：

**基础情感：** (angry 生气) (sad 伤心) (excited 兴奋) (surprised 惊讶) (satisfied 满意) (delighted 高兴) (scared 害怕) (worried 担心) (upset 沮丧) (nervous 紧张) (frustrated 挫败) (depressed 郁闷) (empathetic 同情) (embarrassed 尴尬) (disgusted 厌恶) (moved 感动) (proud 自豪) (relaxed 放松) (grateful 感激) (confident 自信) (interested 感兴趣) (curious 好奇) (confused 困惑) (joyful 快乐)

**高级情感：** (disdainful 鄙视) (unhappy 不开心) (anxious 焦虑) (hysterical 歇斯底里) (indifferent 冷漠) (impatient 不耐烦) (guilty 内疚) (scornful 轻蔑) (panicked 恐慌) (furious 愤怒) (reluctant 不情愿) (keen 热衷) (disapproving 不赞成) (negative 消极) (denying 否认) (astonished 震惊) (serious 严肃) (sarcastic 讽刺) (conciliative 安抚) (comforting 安慰) (sincere 真诚) (sneering 冷笑) (hesitating 犹豫) (yielding 屈服) (painful 痛苦) (awkward 尴尬) (amused 觉得有趣)

**语调标记：** (in a hurry tone 急促的语调) (shouting 喊叫) (screaming 尖叫) (whispering 耳语) (soft tone 柔和的语调)

**特殊音频效果：** (laughing 笑声) (chuckling 轻笑) (sobbing 抽泣) (crying loudly 大声哭泣) (sighing 叹息) (panting 喘息) (groaning 呻吟) (crowd laughing 人群笑声) (background laughter 背景笑声) (audience laughing 观众笑声)

**使用提示：** 您也可以使用 "哈哈哈" 来控制笑声，还有许多其他用法等待您自己探索。
"""

TEXTBOX_PLACEHOLDER = "请在此输入您的文字。使用情感标记如 (happy 开心), (sad 伤心), (excited 兴奋) 来控制语音的语调和情感。"

# Example texts with emotion controls
EXAMPLES = [
    # Basic emotions examples
    ["Hello everyone! (excited) I'm absolutely thrilled to be here today and share this incredible journey with all of you. This is going to be an amazing experience that we'll remember for years to come!", "Basic Emotions - Excited"],
    ["I'm really (sad) disappointed and heartbroken about what happened yesterday. It's been weighing heavily on my mind, and I can't stop thinking about how different things could have been if we had made other choices.", "Basic Emotions - Sad"],
    ["(angry) I can't believe this is happening again! We've been through this exact same situation three times before, and yet here we are, making the same mistakes over and over. This is absolutely unacceptable!", "Basic Emotions - Angry"],
    ["(surprised) Wow! I never expected this to happen in a million years! When I woke up this morning, this was the last thing on my mind. Life really has a way of surprising us when we least expect it.", "Basic Emotions - Surprised"],
    ["(relaxed) Take your time, there's absolutely no need to rush or feel pressured. We have all the time in the world to get this right. Let's just breathe, stay calm, and approach this step by step.", "Basic Emotions - Relaxed"],
    ["(grateful) Thank you so much for your incredible help and unwavering support throughout this challenging period. I honestly don't know how I would have managed without your kindness and dedication. You've truly made all the difference.", "Basic Emotions - Grateful"],
    ["(nervous) I'm feeling quite anxious about the presentation tomorrow. My hands are shaking just thinking about standing in front of all those important people. What if I forget my lines or make a terrible mistake?", "Basic Emotions - Nervous"],
    ["(joyful) This is the happiest day of my entire life! Everything has worked out perfectly, and I feel like I'm floating on cloud nine. All the hard work and patience has finally paid off in the most wonderful way!", "Basic Emotions - Joyful"],
    
    # Advanced emotions examples
    ["(sarcastic) Oh, that's just absolutely wonderful news, isn't it? I'm sure everyone is simply thrilled to hear about yet another delay in the project that was supposed to be completed months ago.", "Advanced Emotions - Sarcastic"],
    ["(sincere) I truly appreciate everything you've done for me and my family during this difficult time. Your genuine kindness and selfless actions have touched our hearts in ways that words simply cannot express.", "Advanced Emotions - Sincere"],
    ["(anxious) I'm really worried sick about the test results that should arrive any day now. The waiting is absolutely killing me, and I can't seem to focus on anything else. What if the news isn't what we're hoping for?", "Advanced Emotions - Anxious"],
    ["(comforting) Everything is going to be perfectly okay, I promise you that. Don't worry about it for another second. We'll get through this together, just like we always do. You're stronger than you know, and I believe in you completely.", "Advanced Emotions - Comforting"],
    ["(hesitating) Well, I'm not really sure if that's the right choice to make in this particular situation. There are so many factors to consider, and I keep going back and forth between the different options. Maybe we should take more time to think this through?", "Advanced Emotions - Hesitating"],
    ["(furious) I am absolutely livid right now! How dare they treat us with such complete disrespect and disregard for everything we've worked so hard to build! This is the final straw, and I won't tolerate it for one more second!", "Advanced Emotions - Furious"],
    ["(empathetic) I can only imagine how incredibly difficult this must be for you right now. Please know that you're not alone in this struggle, and that it's completely normal to feel overwhelmed by everything that's happening.", "Advanced Emotions - Empathetic"],
    
    # Tone markers examples
    ["(whispering) Can you please keep this information strictly between us? It's extremely sensitive, and if word gets out, it could cause serious problems for everyone involved. I'm trusting you with something very important here.", "Tone Markers - Whispering"],
    ["(shouting) Hey! Over here! Can you hear me from all the way over there? I've been trying to get your attention for the past five minutes! The building is on fire and we need to evacuate immediately!", "Tone Markers - Shouting"],
    ["(soft tone) Good night, my dear. Sweet dreams, and may tomorrow bring you all the happiness and peace that your beautiful heart deserves. Sleep well, and know that you are loved beyond measure.", "Tone Markers - Soft"],
    ["(in a hurry tone) Quick! We absolutely must leave right now or we're going to miss the last train of the evening! Grab your jacket and keys, don't worry about anything else. We can figure out the details later!", "Tone Markers - Hurry"],
    ["(screaming) Help me! Somebody please help me! I'm trapped and I can't get out on my own! Is there anyone out there who can hear me? Please, I'm really scared and I need assistance right away!", "Tone Markers - Screaming"],
    
    # Special audio effects examples
    ["That joke was absolutely hilarious and caught me completely off guard! (laughing) Ha, ha, ha! I haven't laughed that hard in years! You really have a wonderful sense of humor and perfect timing!", "Audio Effects - Laughing"],
    ["(sighing) I suppose we'll have to try again tomorrow and hope for better results. It's been such a long and exhausting day, and nothing seems to be going according to our carefully laid plans.", "Audio Effects - Sighing"],
    ["I'm completely exhausted after that incredibly intense workout session at the gym. (panting) I pushed myself harder than ever before, but it feels amazing to know that I'm getting stronger every single day.", "Audio Effects - Panting"],
    ["(chuckling) You're quite the interesting character, aren't you? I never know what you're going to say or do next, and that's exactly what makes spending time with you so entertaining and unpredictable.", "Audio Effects - Chuckling"],
    ["(sobbing) I just can't stop crying about what happened. It's been hours since I got the news, but the tears just keep coming. I don't think I'll ever be able to get over this terrible loss.", "Audio Effects - Sobbing"],
    ["The audience was absolutely delighted by the comedian's performance. (audience laughing) Everyone was having such a wonderful time, and the energy in the room was simply electric and contagious.", "Audio Effects - Audience Laughing"],
    
    # Chinese examples
    ["大家好！(兴奋) 今天天气真的非常不错啊！阳光明媚，微风习习，这样的好天气让人心情特别愉快。我觉得我们应该出去走走，享受这美好的一天，不要浪费了这么棒的天气！", "中文情感 - 兴奋"],
    ["(伤心) 我今天心情真的很不好，感觉整个世界都变得灰暗了。昨天发生的事情一直在我脑海里挥之不去，我不知道该如何面对接下来的日子。希望时间能够慢慢治愈这些痛苦吧。", "中文情感 - 伤心"],
    ["(生气) 这件事情真的让我非常非常愤怒！我们已经讨论过很多次了，为什么还是会出现同样的问题？这种不负责任的态度是完全不能接受的！", "中文情感 - 生气"],
    ["(耳语) 这个秘密只有我们两个人知道，千万不能告诉任何其他人。如果消息泄露出去的话，后果会非常严重。我相信你能够为我保守这个重要的秘密。", "中文语调 - 耳语"],
    ["(笑声) 哈哈哈，这个笑话实在是太搞笑了！我很久没有笑得这么开心了！你真的很有幽默感，总是能够在恰当的时候说出最有趣的话来逗大家开心！", "中文音效 - 笑声"],
    ["(感激) 非常感谢您在这段困难时期给予我和我的家人如此大的帮助和支持。您的善良和无私真的深深感动了我们，我们永远不会忘记您的恩情。", "中文情感 - 感激"],
    ["(焦虑) 我真的很担心明天的考试结果。这次考试对我来说太重要了，如果考不好的话，我所有的努力都会白费。现在想起来就觉得心跳加速，手心出汗。", "中文情感 - 焦虑"],
    ["(叹息) 唉，看来我们只能明天再试一次了。今天真的是诸事不顺，什么事情都没有按照预期的那样发展。希望明天会是一个全新的开始吧。", "中文音效 - 叹息"],
    
    # Mixed language examples
    ["Hello everyone! (excited) 大家好! I'm incredibly happy to meet you all today! 今天能够见到各位真的是我的荣幸！Let's work together to create something amazing! 让我们一起努力创造美好的未来吧！", "Mixed Language - Excited"],
    ["(soft tone) Good morning, my dear friends. 早上好，亲爱的朋友们！I hope you have a wonderful and productive day ahead of you. 希望大家今天都能过得充实愉快！May all your dreams come true. 愿你们的梦想都能实现！", "Mixed Language - Soft Tone"],
    ["(worried) I'm really concerned about the current situation. 我对目前的情况感到很担心。We need to be very careful with our next steps. 我们接下来的每一步都要非常小心。This could affect everyone involved. 这可能会影响到所有相关的人员。", "Mixed Language - Worried"],
    ["(laughing) That was absolutely hilarious! 刚才那个真的太好笑了！Ha, ha, ha! 哈哈哈！I can't stop laughing! 我都停不下来了！You always know how to make everyone smile! 你总是知道如何让大家开心！", "Mixed Language - Laughing"],
]


def build_app_with_examples(inference_fct: Callable, theme: str = "light") -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="OpenAudio S1", css="""
        .example-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            color: white;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .generate-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            border-radius: 25px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            padding: 15px 30px;
            transition: all 0.3s ease;
        }
        .example-preview {
            background-color: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-style: italic;
        }
    """) as app:
        
        # Header
        with gr.Row():
            with gr.Column():
                gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        # Main Content - Three Column Layout
        with gr.Row():
            # Left Column - Examples and Text Input
            with gr.Column(scale=4):
                gr.Markdown("## 🎭 示例画廊")
                gr.Markdown("*从我们精心策划的示例中选择，探索不同的情感和语调*")
                
                # Example selection section
                with gr.Row():
                    example_dropdown = gr.Dropdown(
                        choices=[f"{example[1]}" for example in EXAMPLES],
                        label="🎯 选择示例",
                        value=None,
                        interactive=True
                    )

                # Example preview
                example_preview = gr.Markdown(
                    "💡 **提示：** 选择上面的示例可以在此处看到预览",
                    elem_classes=["example-preview"]
                )
                
                with gr.Row():
                    load_example_btn = gr.Button("📝 加载示例", variant="secondary", size="lg")
                    clear_text_btn = gr.Button("🗑️ 清空文本", variant="secondary")

                # Text Input Area
                gr.Markdown("## ✍️ 文本输入")
                text = gr.Textbox(
                    label="输入带有情感标记的文字",
                    placeholder=TEXTBOX_PLACEHOLDER,
                    lines=6,
                    max_lines=20,
                    show_copy_button=True
                )
                
                # Quick Actions
                with gr.Row():
                    with gr.Column():
                        generate = gr.Button(
                            value="🎧 生成语音",
                            variant="primary",
                            size="lg",
                            elem_classes=["generate-btn"]
                        )

            # Middle Column - Audio Output
            with gr.Column(scale=3):
                gr.Markdown("## 🔊 生成的音频")
                
                with gr.Row():
                    error = gr.HTML(
                        label="状态",
                        visible=True,
                    )
                
                audio = gr.Audio(
                    label="您生成的语音",
                    type="numpy",
                    interactive=False,
                    visible=True,
                    show_download_button=True,
                    show_share_button=True
                )
                
                # Audio info
                gr.Markdown("""
                **🎵 音频控制：**
                - 点击播放按钮收听
                - 使用下载按钮保存音频
                - 根据需要调节设备音量
                """)

            # Right Column - Settings
            with gr.Column(scale=3):
                gr.Markdown("## ⚙️ 设置")
                
                with gr.Accordion("🎛️ 高级参数", open=False):
                    chunk_length = gr.Slider(
                        label="迭代提示长度 (0 = 关闭)",
                        minimum=100,
                        maximum=400,
                        value=300,
                        step=8,
                        info="控制文本处理块大小"
                    )

                    max_new_tokens = gr.Slider(
                        label="最大令牌数 (0 = 无限制)",
                        minimum=0,
                        maximum=2048,
                        value=0,
                        step=8,
                        info="每次生成的最大令牌数"
                    )

                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-P",
                            minimum=0.7,
                            maximum=0.95,
                            value=0.8,
                            step=0.01,
                        )

                        temperature = gr.Slider(
                            label="温度",
                            minimum=0.7,
                            maximum=1.0,
                            value=0.8,
                            step=0.01,
                        )

                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="重复惩罚",
                            minimum=1,
                            maximum=1.2,
                            value=1.1,
                            step=0.01,
                        )

                        seed = gr.Number(
                            label="种子 (0 = 随机)",
                            value=0,
                            info="用于可重现的结果"
                        )

                with gr.Accordion("🎤 参考音频", open=False):
                    gr.Markdown("*上传 5-10 秒的参考音频来克隆特定的说话人*")
                    
                    reference_id = gr.Textbox(
                        label="参考 ID",
                        placeholder="留空以使用上传的音频",
                        info="可选：使用保存的参考 ID"
                    )

                    use_memory_cache = gr.Radio(
                        label="内存缓存",
                        choices=["on", "off"],
                        value="on",
                        info="缓存以加快处理速度"
                    )

                    reference_audio = gr.Audio(
                        label="上传参考音频",
                        type="filepath",
                    )
                    
                    reference_text = gr.Textbox(
                        label="参考文本 (可选)",
                        lines=2,
                        placeholder="与参考音频匹配的文本...",
                        info="有助于更好的声音克隆"
                    )

        # Functions for interactivity
        def preview_example(dropdown_value):
            if dropdown_value is None:
                return "💡 **提示：** 选择上面的示例可以在此处看到预览"
            
            for example in EXAMPLES:
                if example[1] == dropdown_value:
                    return f"**预览：** {example[0][:200]}{'...' if len(example[0]) > 200 else ''}"
            return "未找到示例"

        def load_example(dropdown_value):
            if dropdown_value is None:
                return ""
            
            for example in EXAMPLES:
                if example[1] == dropdown_value:
                    return example[0]
            return ""

        def clear_text():
            return ""

        # Event handlers
        example_dropdown.change(
            preview_example,
            inputs=[example_dropdown],
            outputs=[example_preview]
        )

        load_example_btn.click(
            load_example,
            inputs=[example_dropdown],
            outputs=[text]
        )

        clear_text_btn.click(
            clear_text,
            outputs=[text]
        )

        # Generate button
        generate.click(
            inference_fct,
            inputs=[
                text,
                reference_id,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
            ],
            outputs=[audio, error],
            concurrency_limit=1,
        )

    return app


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/openaudio-s1-mini",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/openaudio-s1-mini/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    # Check if MPS or CUDA is available
    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("mps is available, running on mps.")
    elif torch.xpu.is_available():
        args.device = "xpu"
        logger.info("XPU is available, running on XPU.")
    elif not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    # Create the inference engine
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference_engine.inference(
            ServeTTSRequest(
                text="Hello world.",
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
            )
        )
    )

    logger.info("Warming up done, launching the web UI...")

    # Get the inference function with the immutable arguments
    inference_fct = get_inference_wrapper(inference_engine)

    app = build_app_with_examples(inference_fct, args.theme)
    app.launch(server_name="0.0.0.0", share=False)
