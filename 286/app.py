"""
Qwen-Image-2512 Gradio Web 应用
基于 Qwen-Image-2512 模型的图像生成 Web 界面
"""

import gradio as gr
import torch
from diffusers import DiffusionPipeline
import os

# 模型路径
MODEL_PATH = "./checkpoints/Qwen-Image-2512"

# 预设的宽高比选项
ASPECT_RATIOS = {
    "1:1 (1328×1328)": (1328, 1328),
    "16:9 (1664×928)": (1664, 928),
    "9:16 (928×1664)": (928, 1664),
    "4:3 (1472×1104)": (1472, 1104),
    "3:4 (1104×1472)": (1104, 1472),
    "3:2 (1584×1056)": (1584, 1056),
    "2:3 (1056×1584)": (1056, 1584),
}

# 默认负面提示词
DEFAULT_NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

# 示例提示词 - 按功能分类 [功能增强类别, 提示词]
EXAMPLE_PROMPTS = [
    # 人物写实
    [
        "🧑 人物写实 - 精细发丝和自然表情",
        "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.",
    ],
    [
        "🧑 人物写实 - 面部细节和环境背景",
        "A Chinese female college student, around 20 years old, with a very short haircut that conveys a gentle, artistic vibe. Her hair naturally falls to partially cover her cheeks, projecting a tomboyish yet charming demeanor. She has cool-toned fair skin and delicate features, with a slightly shy yet subtly confident expression—her mouth crooked in a playful, youthful smirk. She wears an off-shoulder top, revealing one shoulder, with a well-proportioned figure. The image is framed as a close-up selfie: she dominates the foreground, while the background clearly shows her dormitory—a neatly made bed with white linens on the top bunk, a tidy study desk with organized stationery, and wooden cabinets and drawers. The photo is captured on a smartphone under soft, even ambient lighting, with natural tones, high clarity, and a bright, lively atmosphere full of youthful, everyday energy.",
    ],
    [
        "🧑 人物写实 - 精确姿态语义遵循",
        "An East Asian teenage boy, aged 15–18, with soft, fluffy black short hair and refined facial contours. His large, warm brown eyes sparkle with energy. His fair skin and sunny, open smile convey an approachable, friendly demeanor—no makeup or blemishes. He wears a blue-and-white summer uniform shirt, slightly unbuttoned, made of thin breathable fabric, with black headphones hanging around his neck. His hands are in his pockets, body leaning slightly forward in a relaxed pose, as if engaged in conversation. Behind him lies a summer school playground: lush green grass and a red rubber track in the foreground, blurred school buildings in the distance, a clear blue sky with fluffy white clouds. The bright, airy lighting evokes a joyful, carefree adolescent atmosphere.",
    ],
    [
        "🧑 人物写实 - 年龄特征（皱纹）渲染",
        "An elderly Chinese couple in their 70s in a clean, organized home kitchen. The woman has a kind face and a warm smile, wearing a patterned apron; the man stands behind her, also smiling, as they both gaze at a steaming pot of buns on the stove. The kitchen is bright and tidy, exuding warmth and harmony. The scene is captured with a wide-angle lens to fully show the subjects and their surroundings.",
    ],
    # 自然风景
    [
        "🌿 自然纹理 - 水流植被雾气渲染",
        "A turquoise river winds through a lush canyon. Thick moss and dense ferns blanket the rocky walls; multiple waterfalls cascade from above, enveloped in mist. At noon, sunlight filters through the dense canopy, dappling the river surface with shimmering light. The atmosphere is humid and fresh, pulsing with primal jungle vitality. No humans, text, or artificial traces present.",
    ],
    [
        "🌿 自然纹理 - 海浪与晨雾渲染",
        "At dawn, a thin mist veils the sea. An ancient stone lighthouse stands at the cliff's edge, its beacon faintly visible through the fog. Black rocks are pounded by waves, sending up bursts of white spray. The sky glows in soft blue-purple hues under cool, hazy light—evoking solitude and solemn grandeur.",
    ],
    # 动物毛发
    [
        "🐕 动物毛发 - 精细毛发纹理",
        "An ultra-realistic close-up of a golden retriever outdoors under soft daylight. Hair is exquisitely detailed: strands distinct, color transitioning naturally from warm gold to light cream, light glinting delicately at the tips; a gentle breeze adds subtle volume. Undercoat is soft and dense; guard hairs are long and well-defined, with visible layering. Eyes are moist, expressive; nose is slightly damp with fine specular highlights. Background is softly blurred to emphasize the dog's tangible texture and vivid expression.",
    ],
    [
        "🐕 动物毛发 - 粗糙野生动物纹理",
        "A male argali stands atop a barren, rocky mountainside. Its coarse, dense grey-brown coat covers a powerful, muscular body. Most striking are its massive, thick, outward-spiraling horns—a symbol of wild strength. Its gaze is alert and sharp. The background reveals steep alpine terrain: jagged peaks, sparse low vegetation, and abundant sunlight—conveying the harsh yet majestic wilderness and the animal's resilient vitality.",
    ],
    # 文字渲染
    [
        "📝 文字渲染 - PPT时间轴图文混排",
        '这是一张现代风格的科技感幻灯片，整体采用深蓝色渐变背景。标题是"Qwen-Image发展历程"。下方一条水平延伸的发光时间轴，轴线中间写着"生图路线"。由左侧淡蓝色渐变为右侧深紫色，并以精致的箭头收尾。时间轴上每个节点通过虚线连接至下方醒目的蓝色圆角矩形日期标签，标签内为清晰白色字体，从左向右依次写着："2025年5月6日 Qwen-Image 项目启动""2025年8月4日 Qwen-Image 开源发布""2025年12月31日 Qwen-Image-2512 开源发布" （周围光晕显著）在下方一条水平延伸的发光时间轴，轴线中间写着"编辑路线"。由左侧淡蓝色渐变为右侧深紫色，并以精致的箭头收尾。时间轴上每个节点通过虚线连接至下方醒目的蓝色圆角矩形日期标签，标签内为清晰白色字体，从左向右依次写着："2025年8月18日 Qwen-Image-Edit 开源发布""2025年9月22日 Qwen-Image-Edit-2509 开源发布""2025年12月19日 Qwen-Image-Layered 开源发布""2025年12月23日 Qwen-Image-Edit-2511 开源发布"',
    ],
    [
        "📝 文字渲染 - 产品对比图混合渲染",
        '这是一张现代风格的科技感幻灯片，整体采用深蓝色渐变背景。顶部中央为白色无衬线粗体大字标题"Qwen-Image-2512重磅发布"。画面主体为横向对比图，视觉焦点集中于中间的升级对比区域。左侧为面部光滑没有任何细节的女性人像，质感差；右侧为高度写实的年轻女性肖像，皮肤呈现真实毛孔纹理与细微光影变化，发丝根根分明，眼眸透亮，表情自然，整体质感接近写实摄影。两图像之间以一个绿色流线型箭头链接。造型科技感十足，中部标注"2512质感升级"，使用白色加粗字体，居中显示。箭头两侧有微弱光晕效果，增强动态感。在图像下方，以白色文字呈现三行说明："● 更真实的人物质感。大幅度降低了生成图片的AI感，提升了图像真实性 ● 更细腻的自然纹理。大幅度提升了生成图片的纹理细节。风景图，动物毛发刻画更细腻。● 更复杂的文字渲染。大幅提升了文字渲染的质量。图文混合渲染更准确，排版更好"',
    ],
    [
        "📝 文字渲染 - 工业信息图表复杂布局",
        '这是一幅专业级工业技术信息图表，整体采用深蓝色科技感背景，光线均匀柔和，营造出冷静、精准的现代工业氛围。画面分为左右两大板块，布局清晰，视觉层次分明。左侧板块标题为"实际发生的现象"，以浅蓝色圆角矩形框突出显示，内部排列三个深蓝色按钮式条目，第一个条目展示一堆棕色粉末状原料上滴落水滴的图标，文字为"团聚/结块"，后面配有绿色对钩；第二个条目为一个装有蓝色液体并冒出气泡的锥形瓶，文字为"产生气泡/缺陷"，后面配有绿色对钩；第三个条目为两个生锈的齿轮，文字为"设备腐蚀/催化剂失活"，后面配有绿色对钩。右侧板块标题为"【不会】发生的现象"，使用米黄色圆角矩形框呈现，内部四个条目均置于深灰色背景方框中。图标分别为：一组精密啮合的金属齿轮，文字为"反应效率【显著提高】"，上方覆盖醒目的红色叉号；一捆整齐排列的金属管材，文字为"成品内部【绝对无气泡/孔隙】"，上方覆盖醒目的红色叉号；一条坚固的金属链条正在承受拉力，文字为"材料强度与耐久性【得到增强】"，上方覆盖醒目的红色叉号；一堆腐蚀的扳手，文字为"加工过程【零腐蚀/零副反应风险】"，上方覆盖醒目的红色叉号。底部中央有一行小字注释："注：水分的存在通常会导致负面或干扰性的结果，而非理想或增强的状态"，字体为白色，清晰可读。整体风格现代简约，配色对比强烈，图形符号准确传达技术逻辑，适合用于工业培训或科普演示场景。',
    ],
    [
        "📝 文字渲染 - 网格海报时间标注",
        '这是一幅由十二个分格组成的3×4网格布局的写实摄影作品，整体呈现"健康的一天"主题，画面风格简洁清晰，每一分格独立成景又统一于生活节奏的叙事脉络。第一行分别是"06:00 晨跑唤醒身体"：面部特写，一位女性身穿灰色运动套装，背景是初升的朝阳与葱郁绿树；"06:30 动态拉伸激活关节"：女性身着瑜伽服在阳台做晨间拉伸，身体舒展，背景为淡粉色天空与远山轮廓；"07:30 均衡营养早餐"：桌上摆放全麦面包、牛油果和一杯橙汁，女性微笑着准备用餐；"08:00 补水润燥"：透明玻璃水杯中浮有柠檬片，女性手持水杯轻啜，阳光从左侧斜照入室，杯壁水珠滑落；第二行分别是："09:00 专注高效工作"：女性专注敲击键盘，屏幕显示简洁界面，身旁放有一杯咖啡与一盆绿植；"12:00 静心阅读时光"：女性坐在书桌前翻阅纸质书籍，台灯散发暖光，书页泛黄，旁放半杯红茶；"12:30 午后轻松漫步"：女性在林荫道上漫步，脸部特写；"15:00 茶香伴午后"：女性端着骨瓷茶杯站在窗边，窗外是城市街景与飘动云朵，茶香袅袅；第三行分别是："18:00 运动释放压力"：健身房内，女性正在练习瑜伽；"19:00 美味晚餐"：女性在开放式厨房中切菜，砧板上有番茄与青椒，锅中热气升腾，灯光温暖；"21:00 冥想助眠"：女性盘腿坐在柔软地毯上冥想，双手轻放膝上，闭目宁静；"21:30 进入睡眠"：女性躺在床上休息。整体采用自然光线为主，色调以暖白与米灰为基调，光影层次分明，画面充满温馨的生活气息与规律的节奏感。',
    ],
]

# 全局变量存储模型管道
pipe = None


def load_model():
    """加载 Qwen-Image-2512 模型"""
    global pipe
    
    print("正在加载 Qwen-Image-2512 模型...")
    
    # 检测设备和数据类型
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        torch_dtype = torch.float32
        device = "cpu"
        print("CUDA 不可用，使用 CPU（生成速度会较慢）")
    
    # 加载模型
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch_dtype
    ).to(device)
    
    print("模型加载完成！")
    return pipe


def generate_image(
    prompt: str,
    negative_prompt: str,
    aspect_ratio: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    seed: int,
    progress=gr.Progress()
):
    """生成图像"""
    global pipe
    
    if pipe is None:
        return None, "错误：模型未加载，请刷新页面重试。"
    
    if not prompt.strip():
        return None, "错误：请输入提示词。"
    
    try:
        # 获取宽高
        width, height = ASPECT_RATIOS[aspect_ratio]
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 设置随机种子
        generator = torch.Generator(device=device).manual_seed(seed)
        
        progress(0, desc="开始生成图像...")
        
        # 生成图像
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator
        )
        
        image = result.images[0]
        
        info = f"✅ 生成成功！\n尺寸: {width}×{height}\n步数: {num_inference_steps}\nCFG: {true_cfg_scale}\n种子: {seed}"
        
        return image, info
        
    except Exception as e:
        error_msg = f"❌ 生成失败：{str(e)}"
        print(error_msg)
        return None, error_msg


def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(
        title="Qwen-Image-2512 图像生成"
    ) as demo:
        
        gr.HTML("""
        <div class="main-title">
            <h1>🎨 Qwen-Image-2512 图像生成</h1>
            <p>基于 Qwen-Image-2512 模型的 AI 图像生成工具</p>
            <p style="font-size: 0.9em; color: #666;">特色：增强人物写实感 | 精细自然纹理 | 复杂文字渲染</p>
        </div>
        """)
        
        with gr.Row():
            # 左侧输入区域
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="✏️ 提示词 (Prompt)",
                    placeholder="请输入图像描述...",
                    lines=5,
                    max_lines=10
                )
                
                negative_prompt = gr.Textbox(
                    label="🚫 负面提示词 (Negative Prompt)",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=3,
                    max_lines=5
                )
                
                with gr.Row():
                    aspect_ratio = gr.Dropdown(
                        label="📐 宽高比",
                        choices=list(ASPECT_RATIOS.keys()),
                        value="16:9 (1664×928)"
                    )
                    
                    seed = gr.Number(
                        label="🎲 随机种子",
                        value=42,
                        precision=0
                    )
                
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="🔄 推理步数",
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=1
                    )
                    
                    true_cfg_scale = gr.Slider(
                        label="🎯 CFG 强度",
                        minimum=1.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.5
                    )
                
                generate_btn = gr.Button(
                    "🚀 生成图像",
                    variant="primary",
                    size="lg"
                )
            
            # 右侧输出区域
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="🖼️ 生成结果",
                    type="pil"
                )
                
                output_info = gr.Textbox(
                    label="📊 生成信息",
                    lines=5,
                    interactive=False
                )
        
        # 示例区域
        gr.HTML("""
        <div style="margin-top: 2rem;">
            <h3>📚 示例提示词</h3>
            <p style="color: #666; font-size: 0.9em;">点击下方示例快速体验不同功能场景</p>
        </div>
        """)
        
        # 功能分类标签（仅用于显示，不绑定输入）
        feature_category = gr.Textbox(visible=False)
        
        # 创建示例
        gr.Examples(
            examples=EXAMPLE_PROMPTS,
            inputs=[feature_category, prompt],
            label="",
            examples_per_page=12
        )
        
        # 功能说明
        with gr.Accordion("💡 功能说明", open=False):
            gr.Markdown("""
            ### Qwen-Image-2512 核心增强功能
            
            | 功能类别 | 说明 |
            |---------|------|
            | **🧑 增强人物写实** | 大幅提升面部细节、发丝渲染、年龄特征，降低 AI 感 |
            | **🌿 精细自然纹理** | 水流、植被、雾气、动物毛发等自然元素更加细腻 |
            | **📝 复杂文字渲染** | 支持 PPT、信息图、海报等复杂图文混排场景 |
            
            ### 参数说明
            
            - **提示词**: 描述你想生成的图像内容，越详细越好
            - **负面提示词**: 描述你不想在图像中出现的元素
            - **宽高比**: 选择适合场景的图像比例
            - **推理步数**: 步数越多质量越高，但生成时间越长（推荐 50）
            - **CFG 强度**: 控制图像与提示词的匹配程度（推荐 4.0）
            - **随机种子**: 相同种子 + 相同参数 = 相同图像，方便复现
            """)
        
        # 绑定事件
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, aspect_ratio, num_inference_steps, true_cfg_scale, seed],
            outputs=[output_image, output_info]
        )
    
    return demo


if __name__ == "__main__":
    # 启动时加载模型
    load_model()
    
    # 创建并启动界面
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
