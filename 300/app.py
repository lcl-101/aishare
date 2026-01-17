import torch
import gradio as gr
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image
import os

# 全局模型变量
pipe = None

def load_model():
    """加载模型"""
    global pipe
    if pipe is None:
        print("正在加载 GLM-Image 模型...")
        model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "GLM-Image")
        pipe = GlmImagePipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        print("模型加载完成！")
    return pipe

def text_to_image(prompt, height, width, num_inference_steps, guidance_scale, seed):
    """文生图功能"""
    if not prompt:
        return None, "请输入提示词"
    
    try:
        pipe = load_model()
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        
        # 确保高度和宽度是32的倍数
        height = int(height) // 32 * 32
        width = int(width) // 32 * 32
        
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]
        
        return image, "图片生成成功！"
    except Exception as e:
        return None, f"生成失败: {str(e)}"

def image_to_image(prompt, input_image, height, width, num_inference_steps, guidance_scale, seed):
    """图生图功能"""
    if not prompt:
        return None, "请输入提示词"
    if input_image is None:
        return None, "请上传图片"
    
    try:
        pipe = load_model()
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        
        # 确保高度和宽度是32的倍数
        height = int(height) // 32 * 32
        width = int(width) // 32 * 32
        
        # 转换图片格式
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        elif hasattr(input_image, 'convert'):
            input_image = input_image.convert("RGB")
        
        image = pipe(
            prompt=prompt,
            image=[input_image],
            height=height,
            width=width,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]
        
        return image, "图片编辑成功！"
    except Exception as e:
        return None, f"编辑失败: {str(e)}"

# 文生图示例提示词
t2i_examples = [
    # 双十一活动海报
    ['一张充满活力、引人注目的双十一购物节促销海报设计。主视觉中央是醒目的红金色大字「11.11」，周围环绕着爆炸般的粒子特效和放射状光芒。背景是从深红到橙色的渐变，周围漂浮着购物袋、礼盒和折扣标签。顶部有一条横幅，用优雅的中国书法风格写着「双十一狂欢购」。底部区域展示促销信息：「全场5折」、「包邮」、「限时抢购」。装饰元素包括金色丝带、五彩纸屑和小型烟花效果。整体风格喜庆、奢华且充满活力，具有现代电商美学。配色以红色、金色和白色为主，营造出庆祝和紧迫感。'],
    # 树莓慕斯蛋糕食谱
    ['一幅精心设计的现代美食杂志风格甜点食谱插图，主题是树莓慕斯蛋糕。整体布局干净明亮，分为四个主要区域：左上角是醒目的黑色标题「树莓慕斯蛋糕制作指南」，右侧是柔光拍摄的成品蛋糕特写照片，展示淡粉色蛋糕上装饰着新鲜树莓和薄荷叶；左下角是食材清单区域，标题为「所需食材」，列出「面粉150克」、「鸡蛋3个」、「糖120克」、「树莓果泥200克」、「吉利丁片10克」、「淡奶油300毫升」和「新鲜树莓若干」，每项旁边配有简约线条图标（如面粉袋、鸡蛋、糖罐等）；右下角展示四个大小相等的步骤框，每个框内包含高清微距照片和相应说明。整体色调以奶油白和淡粉色为主。'],
    # 科技公司宣传图
    ['一张未来感十足的科技公司宣传横幅，采用简洁现代的设计风格。中央展示一个全息投影效果的AI大脑图标，周围环绕着电蓝色和青色的发光神经网络连接线。背景是深海军蓝渐变，点缀着精细的几何图案和漂浮的数据粒子。画面上方用现代无衬线字体写着「2024人工智能创新峰会」。主标题下方是较小的文字「塑造明日智慧」。底部角落有抽象的电路板图案和公司Logo。整体美学干净、专业、前瞻，以冷蓝色调为主。'],
    # 咖啡店菜单
    ['一张温馨的手工咖啡店菜单设计，采用复古乡村风格。背景是做旧的羊皮纸质感，带有咖啡渍水印效果。顶部是手绘的冒着热气的咖啡杯插图，配以艺术字体的店名「香醇咖啡屋」。菜单分为几个区域：热饮区列出「浓缩咖啡 ¥18」、「卡布奇诺 ¥28」、「拿铁 ¥28」、「摩卡 ¥32」；冷饮区列出「冰美式 ¥25」、「冷萃咖啡 ¥30」、「星冰乐 ¥35」；点心区列出「可颂 ¥18」、「玛芬蛋糕 ¥22」。每个项目旁边都有小型手绘图标。配色采用温暖的棕色、奶油色，点缀森林绿。'],
    # 音乐节海报
    ['一张动感十足的夏日音乐节海报，充满爆发力。背景是从橙色到紫色的夕阳渐变，底部是欢呼人群的剪影。舞台中央展示抽象的音乐元素：漂浮的黑胶唱片、电吉他和霓虹粉色、黄色的音符。醒目的涂鸦风格文字写着「夏日节拍音乐节」，日期为「2024年7月15-17日」。列出的表演嘉宾包括「DJ雷霆」、「霓虹之光乐队」、「电子梦想组合」。装饰元素包括棕榈树剪影、声波图案和几何孟菲斯风格图形。整体氛围青春、活力、欢庆。'],
]

# 图生图示例提示词
i2i_examples = [
    # 将双十一改为618
    ['将这张双十一购物节海报转换为618年中购物节海报。把所有「11.11」文字改为「6.18」，将「双十一狂欢购」改为「618年中大促」。保持相同的喜庆促销风格和红金色调，但添加一些夏日元素如阳光和清新的绿色点缀，体现六月的时节特点。保留折扣标签、购物元素和庆祝装饰。'],
    # 更换背景
    ['将这张图片的背景替换为美丽的日落海滩场景，有金色的沙滩、平静的海浪，以及从橙色渐变到紫色的绚丽天空。'],
    # 风格转换 - 水彩画
    ['将这张图片转换为精致的水彩画风格，带有柔和的笔触、半透明的色彩晕染效果，以及可见的纸张纹理。在边缘添加微妙的颜色渗透效果。'],
    # 季节变换
    ['将这张图片中的季节从夏天改为冬天。添加覆盖地面和表面的积雪、窗户上的霜花、光秃秃的树枝，以及带有冷蓝色调的寒冷氛围，空气中可见呼出的白气。'],
    # 添加节日装饰
    ['为这个场景添加圣诞装饰：闪烁的彩灯、装饰精美的圣诞树、红色和金色的装饰品、包装好的礼物，以及温暖灯光下的温馨节日氛围。'],
    # 时间变换
    ['将这个白天的场景转换为魔幻的夜晚场景，有繁星点点的天空、发光的月亮、路灯发出的柔和环境光，以及神秘的阴影效果。'],
]

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="GLM-Image 图像生成", theme=gr.themes.Soft()) as demo:
        # YouTube频道信息
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">🎬 AI 技术分享频道</h2>
            <p style="color: white; margin: 10px 0;">欢迎访问我的 YouTube 频道，获取更多 AI 技术教程和分享！</p>
            <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" 
               style="display: inline-block; padding: 10px 20px; background-color: #ff0000; color: white; 
                      text-decoration: none; border-radius: 5px; font-weight: bold;">
                📺 访问 YouTube 频道
            </a>
        </div>
        """)
        
        gr.Markdown("# 🎨 GLM-Image 图像生成工具")
        gr.Markdown("基于 GLM-Image 模型的智能图像生成与编辑工具")
        
        with gr.Tabs() as tabs:
            # 第一个Tab：文生图
            with gr.TabItem("🖼️ 文生图", id=0):
                gr.Markdown("### 输入文字描述，生成精美图片")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        t2i_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入图片描述...",
                            lines=5
                        )
                        
                        with gr.Row():
                            t2i_height = gr.Slider(
                                minimum=256, maximum=2048, value=1024, step=32,
                                label="图片高度"
                            )
                            t2i_width = gr.Slider(
                                minimum=256, maximum=2048, value=1152, step=32,
                                label="图片宽度"
                            )
                        
                        with gr.Row():
                            t2i_steps = gr.Slider(
                                minimum=10, maximum=100, value=50, step=1,
                                label="推理步数"
                            )
                            t2i_guidance = gr.Slider(
                                minimum=1.0, maximum=10.0, value=1.5, step=0.1,
                                label="引导强度"
                            )
                        
                        t2i_seed = gr.Number(label="随机种子", value=42)
                        
                        t2i_generate_btn = gr.Button("🎨 生成图片", variant="primary")
                        t2i_status = gr.Textbox(label="状态", interactive=False)
                    
                    with gr.Column(scale=1):
                        t2i_output = gr.Image(label="生成结果", type="pil")
                        t2i_send_btn = gr.Button("📤 发送到图片编辑", variant="secondary")
                
                gr.Markdown("### 📝 示例提示词（点击使用）")
                t2i_example_btns = []
                example_labels = [
                    "🛒 双十一活动海报",
                    "🍰 树莓慕斯蛋糕食谱",
                    "💻 科技公司宣传图",
                    "☕ 咖啡店菜单",
                    "🎵 音乐节海报"
                ]
                
                with gr.Row():
                    for i, label in enumerate(example_labels):
                        btn = gr.Button(label, size="sm")
                        t2i_example_btns.append((btn, i))
            
            # 第二个Tab：图生图
            with gr.TabItem("✏️ 图片编辑", id=1):
                gr.Markdown("### 上传图片，输入编辑指令，智能修改图片")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        i2i_input = gr.Image(
                            label="输入图片",
                            type="pil"
                        )
                        
                        i2i_prompt = gr.Textbox(
                            label="编辑提示词",
                            placeholder="请输入编辑指令...",
                            lines=5
                        )
                        
                        with gr.Row():
                            i2i_height = gr.Slider(
                                minimum=256, maximum=2048, value=1056, step=32,
                                label="输出高度"
                            )
                            i2i_width = gr.Slider(
                                minimum=256, maximum=2048, value=1024, step=32,
                                label="输出宽度"
                            )
                        
                        with gr.Row():
                            i2i_steps = gr.Slider(
                                minimum=10, maximum=100, value=50, step=1,
                                label="推理步数"
                            )
                            i2i_guidance = gr.Slider(
                                minimum=1.0, maximum=10.0, value=1.5, step=0.1,
                                label="引导强度"
                            )
                        
                        i2i_seed = gr.Number(label="随机种子", value=42)
                        
                        i2i_generate_btn = gr.Button("✏️ 编辑图片", variant="primary")
                        i2i_status = gr.Textbox(label="状态", interactive=False)
                    
                    with gr.Column(scale=1):
                        i2i_output = gr.Image(label="编辑结果", type="pil")
                
                gr.Markdown("### 📝 示例编辑提示词（点击使用）")
                i2i_example_btns = []
                i2i_example_labels = [
                    "🎉 双十一改为618活动",
                    "🏖️ 更换为海滩背景",
                    "🎨 转换为水彩画风格",
                    "❄️ 夏天改为冬天",
                    "🎄 添加圣诞装饰",
                    "🌙 白天改为夜晚"
                ]
                
                with gr.Row():
                    for i, label in enumerate(i2i_example_labels):
                        btn = gr.Button(label, size="sm")
                        i2i_example_btns.append((btn, i))
        
        # 文生图事件绑定
        t2i_generate_btn.click(
            fn=text_to_image,
            inputs=[t2i_prompt, t2i_height, t2i_width, t2i_steps, t2i_guidance, t2i_seed],
            outputs=[t2i_output, t2i_status]
        )
        
        # 发送到编辑器 - 直接传递图片并切换Tab
        def copy_image_to_editor(image):
            return image
        
        t2i_send_btn.click(
            fn=copy_image_to_editor,
            inputs=[t2i_output],
            outputs=[i2i_input]
        ).then(
            fn=lambda: gr.Tabs(selected=1),
            inputs=None,
            outputs=[tabs]
        )
        
        # 文生图示例按钮事件
        for btn, idx in t2i_example_btns:
            btn.click(
                fn=lambda i=idx: t2i_examples[i][0],
                inputs=[],
                outputs=[t2i_prompt]
            )
        
        # 图生图事件绑定
        i2i_generate_btn.click(
            fn=image_to_image,
            inputs=[i2i_prompt, i2i_input, i2i_height, i2i_width, i2i_steps, i2i_guidance, i2i_seed],
            outputs=[i2i_output, i2i_status]
        )
        
        # 图生图示例按钮事件
        for btn, idx in i2i_example_btns:
            btn.click(
                fn=lambda i=idx: i2i_examples[i][0],
                inputs=[],
                outputs=[i2i_prompt]
            )
        
        gr.Markdown("""
        ---
        ### 💡 使用说明
        
        **文生图功能：**
        1. 在提示词框中输入详细的图片描述
        2. 调整图片尺寸和生成参数
        3. 点击"生成图片"按钮
        4. 生成完成后可以点击"发送到图片编辑"进行进一步修改
        
        **图片编辑功能：**
        1. 上传需要编辑的图片（或从文生图发送过来）
        2. 输入编辑指令描述你想要的修改
        3. 调整输出尺寸和生成参数
        4. 点击"编辑图片"按钮
        
        **提示：** 提示词越详细，生成效果越好！
        """)
    
    return demo

if __name__ == "__main__":
    # 启动时预加载模型
    print("正在预加载模型...")
    load_model()
    print("模型预加载完成，启动 Web 界面...")
    
    # 创建并启动界面
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
