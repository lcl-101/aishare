import gradio as gr
import numpy as np
import random
import os
import time
import torch
from PIL import Image
from diffusers import DiffusionPipeline

# 配置参数
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1440

# 全局变量存储管道
pipe = None

def load_model():
    """加载Qwen-Image模型"""
    global pipe
    if pipe is None:
        model_name = "checkpoints/Qwen-Image"
        
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"
        
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        print(f"模型已加载到 {device}")
    return pipe

# 预设的提示词示例
TEXT_GENERATION_EXAMPLES = [
    ["宫崎骏的动漫风格。平视角拍摄，阳光下的古街热闹非凡。一个穿着青衫、手里拿着写着'阿里云'卡片的逍遥派弟子站在中间。旁边两个小孩惊讶的看着他。左边有一家店铺挂着'云存储'的牌子，里面摆放着发光的服务器机箱，门口两个侍卫守护者。右边有两家店铺，其中一家挂着'云计算'的牌子，一个穿着旗袍的美丽女子正看着里面闪闪发光的电脑屏幕；另一家店铺挂着'云模型'的牌子，门口放着一个大酒缸，上面写着'千问'，一位老板娘正在往里面倒发光的代码溶液。", "", 42, 1408, 928, 4.0, 50],
    ["一副典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联上左书'义本生知人机同道善思新'，右书'通云赋智乾坤启数高志远'， 横批'智启通义'，字体飘逸，中间挂在一着一副中国风的画作，内容是岳阳楼。", "", 123, 1328, 1328, 4.0, 50],
    ["Bookstore window display. A sign displays 'New Arrivals This Week'. Below, a shelf tag with the text 'Best-Selling Novels Here'. To the side, a colorful poster advertises 'Author Meet And Greet on Saturday' with a central portrait of the author. There are four books on the bookshelf, namely 'The light between worlds' 'When stars are scattered' 'The slient patient' 'The night circus'", "", 456, 928, 1408, 4.0, 50],
    ["A man in a suit is standing in front of the window, looking at the bright moon outside the window. The man is holding a yellowed paper with handwritten words on it: 'A lantern moon climbs through the silver night, Unfurling quiet dreams across the sky, Each star a whispered promise wrapped in light, That dawn will bloom, though darkness wanders by.' There is a cute cat on the windowsill.", "", 789, 1408, 928, 4.0, 50],
    ["一个穿着'QWEN'标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 '一、Qwen-Image的技术路线： 探索视觉生成基础模型的极限，开创理解与生成一体化的未来。二、Qwen-Image的模型特色：1、复杂文字渲染。支持中英渲染、自动布局； 2、精准图像编辑。支持文字编辑、物体增减、风格变换。三、Qwen-Image的未来愿景：赋能专业内容创作、助力生成式AI发展。'", "", 999, 1328, 1328, 4.0, 50],
    ["A movie poster. The first row is the movie title, which reads 'Imagination Unleashed'. The second row is the movie subtitle, which reads 'Enter a world beyond your imagination'. The third row reads 'Cast: Qwen-Image'. The fourth row reads 'Director: The Collective Imagination of Humanity'. The central visual features a sleek, futuristic computer from which radiant colors, whimsical creatures, and dynamic, swirling patterns explosively emerge, filling the composition with energy, motion, and surreal creativity. The background transitions from dark, cosmic tones into a luminous, dreamlike expanse, evoking a digital fantasy realm. At the bottom edge, the text 'Launching in the Cloud, August 2025' appears in bold, modern sans-serif font with a glowing, slightly transparent effect, evoking a high-tech, cinematic aesthetic. The overall style blends sci-fi surrealism with graphic design flair—sharp contrasts, vivid color grading, and layered visual depth—reminiscent of visionary concept art and digital matte painting, 32K resolution, ultra-detailed.", "", 111, 1408, 928, 4.0, 50],
    ["一张企业级高质量PPT页面图像，整体采用科技感十足的星空蓝为主色调，背景融合流动的发光科技线条与微光粒子特效，营造出专业、现代且富有信任感的品牌氛围；页面顶部左侧清晰展示橘红色Alibaba标志，色彩鲜明、辨识度高。主标题位于画面中央偏上位置，使用大号加粗白色或浅蓝色字体写着'通义千问视觉基础模型'，字体现代简洁，突出技术感；主标题下方紧接一行楷体中文文字：'原生中文·复杂场景·自动布局'，字体柔和优雅，形成科技与人文的融合。下方居中排布展示了四张与图片，分别是：一幅写实与水墨风格结合的梅花特写，枝干苍劲、花瓣清雅，背景融入淡墨晕染与飘雪效果，体现坚韧不拔的精神气质；上方写着黑色的楷体'梅傲'。一株生长于山涧石缝中的兰花，叶片修长、花朵素净，搭配晨雾缭绕的自然环境，展现清逸脱俗的文人风骨；上方写着黑色的楷体'兰幽'。一组迎风而立的翠竹，竹叶随风摇曳，光影交错，背景为青灰色山岩与流水，呈现刚柔并济、虚怀若谷的文化意象；上方写着黑色的楷体'竹清'。一片盛开于秋日庭院的菊花丛，花色丰富、层次分明，配以落叶与古亭剪影，传递恬然自适的生活哲学；上方写着黑色的楷体'菊淡'。所有图片采用统一尺寸与边框样式，呈横向排列。页面底部中央用楷体小字写明'2025年8月，敬请期待'，排版工整、结构清晰，整体风格统一且细节丰富，极具视觉冲击力与品牌调性。", "", 222, 1328, 1328, 4.0, 50]
]



def generate_image(prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps, progress=gr.Progress()):
    """图片生成函数"""
    if not prompt:
        return None, "请输入提示词"
    
    progress(0.1, desc="正在初始化模型...")
    model = load_model()
    
    # 检测语言并添加相应的魔法词
    if any('\u4e00' <= char <= '\u9fff' for char in prompt):
        # 包含中文字符
        positive_magic = "超清，4K，电影级构图。"
    else:
        # 英文提示词
        positive_magic = "Ultra HD, 4K, cinematic composition. "
    
    full_prompt = positive_magic + prompt
    
    try:
        progress(0.3, desc="正在生成图片...")
        
        if seed == -1:
            seed = random.randint(0, MAX_SEED)
        
        generator = torch.Generator(device=model.device).manual_seed(seed)
        
        image = model(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        progress(1.0, desc="完成!")
        
        # 保存图片
        timestamp = int(time.time())
        filename = f"generated_{timestamp}.png"
        image.save(filename)
        
        return image, f"图片生成成功! 种子值: {seed}"
        
    except Exception as e:
        return None, f"生成失败: {str(e)}"



def randomize_seed():
    """随机种子生成"""
    return random.randint(0, MAX_SEED)

def quick_preset(preset_type):
    """快速预设功能"""
    presets = {
        "portrait": {
            "width": 928,
            "height": 1408,
            "guidance": 4.0,
            "steps": 50
        },
        "landscape": {
            "width": 1408,
            "height": 928,
            "guidance": 4.0,
            "steps": 50
        },
        "square": {
            "width": 1328,
            "height": 1328,
            "guidance": 4.0,
            "steps": 50
        },
        "fast": {
            "width": 1328,
            "height": 1328,
            "guidance": 3.0,
            "steps": 25
        },
        "quality": {
            "width": 1328,
            "height": 1328,
            "guidance": 5.0,
            "steps": 75
        }
    }
    return presets.get(preset_type, presets["square"])

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="Qwen-Image WebUI", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #2E86AB; font-size: 2.5em;">🎨 Qwen-Image WebUI</h1>
            <p style="font-size: 1.2em; color: #555;">智能视觉创作的基础模型</p>
            <p style="color: #777;">🖼️ 文本生成图片</p>
            <p style="color: #888; font-size: 0.9em;">适用于YouTube演示 - 展示AI图像生成的无限可能</p>
            <p style="color: #ff6b6b; font-size: 0.85em;">⚠️ 图片编辑、图片理解、风格转换功能即将发布，敬请期待！</p>
        </div>
        """)
        
        # Tab 1: 文本生成图片
        with gr.Tab("📝 文本生成图片", elem_id="text_to_image"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="提示词 (Prompt)",
                        placeholder="描述你想要生成的图片...",
                        lines=3
                    )
                    negative_prompt_input = gr.Textbox(
                        label="负面提示词 (Negative Prompt)",
                        placeholder="描述你不想要的元素...",
                        lines=2
                    )
                    
                    with gr.Row():
                        seed_input = gr.Number(
                            label="种子值 (Seed)",
                            value=42,
                            minimum=-1,
                            maximum=MAX_SEED
                        )
                        random_seed_btn = gr.Button("🎲 随机", size="sm")
                    
                    with gr.Row():
                        width_input = gr.Slider(
                            label="宽度",
                            minimum=512,
                            maximum=MAX_IMAGE_SIZE,
                            value=1408,
                            step=64
                        )
                        height_input = gr.Slider(
                            label="高度",
                            minimum=512,
                            maximum=MAX_IMAGE_SIZE,
                            value=928,
                            step=64
                        )
                    
                    # 快速预设按钮
                    with gr.Row():
                        gr.Markdown("**快速预设:**")
                    with gr.Row():
                        portrait_btn = gr.Button("📱 竖屏", size="sm")
                        landscape_btn = gr.Button("🖼️ 横屏", size="sm")
                        square_btn = gr.Button("⬜ 方形", size="sm")
                        fast_btn = gr.Button("⚡ 快速", size="sm")
                        quality_btn = gr.Button("💎 高质量", size="sm")
                    
                    with gr.Row():
                        guidance_scale_input = gr.Slider(
                            label="引导尺度",
                            minimum=1.0,
                            maximum=10.0,
                            value=4.0,
                            step=0.1
                        )
                        num_inference_steps_input = gr.Slider(
                            label="推理步数",
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5
                        )
                    
                    generate_btn = gr.Button("🎨 生成图片", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_image = gr.Image(label="生成结果", type="pil")
                    output_info = gr.Textbox(label="生成信息", interactive=False)
            
            # 示例
            gr.Examples(
                examples=TEXT_GENERATION_EXAMPLES,
                inputs=[prompt_input, negative_prompt_input, seed_input, width_input, height_input, guidance_scale_input, num_inference_steps_input],
                outputs=[output_image, output_info],
                fn=generate_image,
                cache_examples=False
            )
        
        # 事件绑定
        random_seed_btn.click(fn=randomize_seed, outputs=seed_input)
        
        # 快速预设按钮事件
        portrait_btn.click(
            fn=lambda: (928, 1408, 4.0, 50),
            outputs=[width_input, height_input, guidance_scale_input, num_inference_steps_input]
        )
        landscape_btn.click(
            fn=lambda: (1408, 928, 4.0, 50),
            outputs=[width_input, height_input, guidance_scale_input, num_inference_steps_input]
        )
        square_btn.click(
            fn=lambda: (1328, 1328, 4.0, 50),
            outputs=[width_input, height_input, guidance_scale_input, num_inference_steps_input]
        )
        fast_btn.click(
            fn=lambda: (1328, 1328, 3.0, 25),
            outputs=[width_input, height_input, guidance_scale_input, num_inference_steps_input]
        )
        quality_btn.click(
            fn=lambda: (1328, 1328, 5.0, 75),
            outputs=[width_input, height_input, guidance_scale_input, num_inference_steps_input]
        )
        
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, negative_prompt_input, seed_input, width_input, height_input, guidance_scale_input, num_inference_steps_input],
            outputs=[output_image, output_info]
        )
        
        # 添加CSS样式
        demo.css = """
        #text_to_image {
            min-height: 600px;
        }
        .gradio-container {
            max-width: 1400px !important;
        }
        .gr-button {
            transition: all 0.3s ease;
        }
        .gr-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .gr-form {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
        }
        """
    
    return demo

if __name__ == "__main__":
    # 预加载模型
    print("正在预加载Qwen-Image模型...")
    load_model()
    print("模型加载完成!")
    
    # 启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
