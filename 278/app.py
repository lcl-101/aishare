import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
import gradio as gr

# 模型路径
MODEL_PATH = "./checkpoints/Qwen-Image-Edit-2511"

# 全局变量存储 pipeline
pipeline = None

# 示例提示词
EXAMPLE_PROMPTS = {
    "character_consistency": [
        {
            "name": "🎄 圣诞节美少女",
            "prompt": "生成圣诞节主题，一位纯欲气质的美少女，图中人脸不变。松散的双麻花辫松散低扎（麻花辫上有布艺彩球装饰），少女气质，无辜眼神，头戴圣诞树造型发饰，小型锥形圣诞树整齐地固定在头顶，顶部是金色五角星，树身装饰着彩色灯串、金色铃铛、蝴蝶结、红蓝金小球，布置精致饱满；冷白皮，白嫩嫩的皮肤如琼玉般嫩滑，纯欲朦胧滤镜，红棕系眼影自然晕染，双手拿着圣诞老人玩偶，圣诞氛围拉满，庆祝感眼神和表情，轻轻歪头，俏皮又好看的动作，可爱与性感并存，反差；蓬松微乱发丝与头顶圣诞树自然融合；穿毛绒红色上衣，质感柔软蓬松；暖白背景、棚拍柔光、低对比度、低饱和度、细腻胶片颗粒、轻微色散光晕、胶片柔光感、温暖治愈氛围、独特视角，非常规构图，70mm胶片人像风格绿色涂鸦描边人物轮廓，描边周围空白处还有各种圣诞节元素的可爱涂鸦，充满童趣和圣诞氛围的手绘拼贴感。人物轮廓荧光红绿金色虚线波点包裹，写满了\"MERRY CHRISMAS\"可爱字体，中景"
        },
        {
            "name": "📸 四宫格表情包",
            "prompt": "生成一张四宫格图片。以下要求：人物：参考图人物分四个画面呈现不同动作表情。左上：双手举过头顶比双\"V\"，眼睛大睁、嘴巴张开，露出惊讶活泼的神态。右上：双手托住脸颊，双眼微闭、嘴巴嘟起，脸颊带红晕，呈现可爱娇憨感。左下：头微侧，一只眼睛wink，舌头吐出，单手比\"V\"，俏皮搞怪。右下：双臂交叉在胸前，眉头微皱、嘴巴嘟起，呈现小傲娇神态。服饰：根据参考图不变。背景与风格：充满疯狂动物城等可爱卡通元素的彩色背景，整体为二次元动漫风格，画面色彩鲜艳、风格甜美治愈，每幅小图都有精致的卡通边框装饰，充满童趣感。"
        },
        {
            "name": "🎨 真人与卡通壁画合影",
            "prompt": "生成竖版3:4画面比例的\"真人与其对应卡通壁画合影\"场景图像：将上传的真实人物照片以原样保留服装、发型、妆容置于画面左侧/前方。在真人背后墙面绘制1:1对应卡通壁画，厚涂质感且采用动漫风格大眼、柔和轮廓五官，完整复刻发型、服装及配饰细节如耳环、项链等，色彩饱和度高并带有涂鸦式笔触效果。墙面添加彩色涂鸦爱心、笑脸、几何图案元素，地面点缀飞溅颜料装饰细节，壁画区域融入如\"2026发财\"的中文字元素，字体风格契合涂鸦美学。确保真人与壁画比例、角度自然衔接，光照方向统一符合场景逻辑，保持整体色彩风格一致，呈现生动、连贯且视觉和谐效果"
        },
        {
            "name": "🎮 像素拼豆成品",
            "prompt": "生成一个手拿着压制好的边缘不规则的平面像素拼豆成品照片，拼豆的内容是参考图中的像素Q版形象，拼豆扁平没有凸起，保持参考图主体特征不变，背景是工作台台面"
        }
    ],
    "multi_person": [
        {
            "name": "🤫 双人嘘手势",
            "prompt": "两个人，一起做一个'嘘'的手势。"
        },
    ],
    "lora": [
        {
            "name": "💡 柔光重照明",
            "prompt": "柔光,使用柔和光线对图片进行重新照明"
        },
        {
            "name": "🔄 镜头左旋30度",
            "prompt": "将镜头向左旋转30度"
        }
    ],
    "industrial_design": [
        {
            "name": "🪖 头盔变银灰色",
            "prompt": "把头盔变成银灰色"
        }
    ]
}

def load_model():
    """加载模型"""
    global pipeline
    if pipeline is None:
        print("正在加载模型...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16
        )
        pipeline.to('cuda')
        pipeline.set_progress_bar_config(disable=None)
        print("模型加载完成!")
    return pipeline

def generate_image(
    image1, 
    image2, 
    prompt, 
    negative_prompt,
    seed,
    true_cfg_scale,
    guidance_scale,
    num_inference_steps,
    num_images_per_prompt
):
    """生成图片"""
    global pipeline
    
    if pipeline is None:
        load_model()
    
    # 处理输入图片
    images = []
    if image1 is not None:
        images.append(Image.fromarray(image1).convert("RGB"))
    if image2 is not None:
        images.append(Image.fromarray(image2).convert("RGB"))
    
    if len(images) == 0:
        raise gr.Error("请至少上传一张图片!")
    
    if not prompt or prompt.strip() == "":
        raise gr.Error("请输入提示词!")
    
    # 设置随机种子
    generator = torch.manual_seed(seed)
    
    # 构建输入参数
    inputs = {
        "image": images if len(images) > 1 else images[0],
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt if negative_prompt else " ",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": num_images_per_prompt,
    }
    
    # 生成图片
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_images = output.images
    
    return output_images

def create_tab_content(tab_name, example_prompts=None):
    """创建通用的Tab内容"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📷 输入图片")
            image1_input = gr.Image(label="图片 1 (必需)", type="numpy")
            image2_input = gr.Image(label="图片 2 (可选)", type="numpy")
            
            gr.Markdown("### ✏️ 提示词")
            prompt_input = gr.Textbox(
                label="提示词 (Prompt)",
                placeholder="描述你想要生成的图片效果...",
                lines=5
            )
            
            # 如果有示例提示词，添加下拉选择
            if example_prompts:
                prompt_choices = ["-- 选择示例提示词 --"] + [p["name"] for p in example_prompts]
                prompt_dropdown = gr.Dropdown(
                    choices=prompt_choices,
                    label="📝 示例提示词模板",
                    value="-- 选择示例提示词 --"
                )
                
                def update_prompt(selected):
                    if selected == "-- 选择示例提示词 --":
                        return ""
                    for p in example_prompts:
                        if p["name"] == selected:
                            return p["prompt"]
                    return ""
                
                prompt_dropdown.change(
                    fn=update_prompt,
                    inputs=[prompt_dropdown],
                    outputs=[prompt_input]
                )
            
            negative_prompt_input = gr.Textbox(
                label="负面提示词 (Negative Prompt)",
                placeholder="描述你不想要的内容...",
                lines=2,
                value=" "
            )
            
            gr.Markdown("### ⚙️ 参数设置")
            with gr.Row():
                seed_input = gr.Number(label="随机种子", value=0, precision=0)
                num_images_input = gr.Slider(
                    label="生成数量", 
                    minimum=1, 
                    maximum=4, 
                    step=1, 
                    value=1
                )
            
            with gr.Row():
                true_cfg_scale_input = gr.Slider(
                    label="True CFG Scale", 
                    minimum=1.0, 
                    maximum=10.0, 
                    step=0.5, 
                    value=4.0
                )
                guidance_scale_input = gr.Slider(
                    label="Guidance Scale", 
                    minimum=0.0, 
                    maximum=5.0, 
                    step=0.1, 
                    value=1.0
                )
            
            num_steps_input = gr.Slider(
                label="推理步数 (Inference Steps)", 
                minimum=10, 
                maximum=100, 
                step=5, 
                value=40
            )
            
            generate_btn = gr.Button("🚀 生成图片", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ 生成结果")
            output_gallery = gr.Gallery(
                label="生成的图片",
                show_label=False,
                columns=1,
                rows=1,
                height=None,
                object_fit="scale-down",
                preview=True
            )
    
    # 绑定生成按钮事件
    generate_btn.click(
        fn=generate_image,
        inputs=[
            image1_input,
            image2_input,
            prompt_input,
            negative_prompt_input,
            seed_input,
            true_cfg_scale_input,
            guidance_scale_input,
            num_steps_input,
            num_images_input
        ],
        outputs=output_gallery
    )
    
    return image1_input, image2_input, prompt_input, output_gallery

# 创建 Gradio 界面
with gr.Blocks(title="Qwen Image Edit 2511", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 Qwen Image Edit 2511
    
    基于 Qwen-Image-Edit-2511 模型的图片编辑工具。上传1-2张图片，输入提示词，即可生成新图片。
    """)
    
    with gr.Tabs():
        # Tab 1: 角色一致性增强
        with gr.TabItem("🎭 角色一致性增强"):
            gr.Markdown("""
            ### 功能说明
            保持人物角色的一致性，支持生成同一角色的不同场景、表情、动作等变化。
            上传参考人物图片，选择或输入提示词，生成保持角色特征的新图片。
            """)
            create_tab_content("character_consistency", EXAMPLE_PROMPTS["character_consistency"])
        
        # Tab 2: 多人合照一致性
        with gr.TabItem("👥 多人合照一致性"):
            gr.Markdown("""
            ### 功能说明
            支持多人合照场景，保持每个人物的特征一致性。
            上传多张人物参考图，生成自然的多人合照效果。
            """)
            create_tab_content("multi_person", EXAMPLE_PROMPTS["multi_person"])
        
        # Tab 3: 内置 LoRA
        with gr.TabItem("🔧 内置 LoRA"):
            gr.Markdown("""
            ### 功能说明
            本模型已内置集成社区流行的 LoRA 能力，包括**光照增强**、**视角变换**等，无需额外加载，通过提示词即可直接使用。
            
            - 💡 **光照增强**: 真实光照控制，可调整图片光线效果
            - 🎬 **视角变换**: 生成新的视角/镜头角度，如平移、旋转等
            """)
            create_tab_content("lora", EXAMPLE_PROMPTS["lora"])
        
        # Tab 4: 工业设计应用
        with gr.TabItem("🏭 工业设计应用"):
            gr.Markdown("""
            ### 功能说明
            适用于工业设计场景，支持产品渲染、材质变换、设计迭代等应用。
            上传产品设计图，生成不同材质、颜色、场景的效果图。
            """)
            create_tab_content("industrial_design", EXAMPLE_PROMPTS["industrial_design"])

if __name__ == "__main__":
    # 预加载模型
    print("正在预加载模型...")
    load_model()
    print("启动 Gradio 服务...")
    
    # 启动 Gradio 服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
