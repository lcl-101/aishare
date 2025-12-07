import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline, LongCatImageEditPipeline

# 全局变量存储已加载的模型
t2i_pipe = None
edit_pipe = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径
T2I_CHECKPOINT_DIR = './checkpoints/LongCat-Image'
EDIT_CHECKPOINT_DIR = './checkpoints/LongCat-Image-Edit'


def load_t2i_pipeline():
    """加载文生图模型"""
    global t2i_pipe
    if t2i_pipe is None:
        print("Loading Text-to-Image pipeline...")
        text_processor = AutoProcessor.from_pretrained(T2I_CHECKPOINT_DIR, subfolder='tokenizer')
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            T2I_CHECKPOINT_DIR,
            subfolder='transformer',
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
        
        t2i_pipe = LongCatImagePipeline.from_pretrained(
            T2I_CHECKPOINT_DIR,
            transformer=transformer,
            text_processor=text_processor
        )
        t2i_pipe.to(device, torch.bfloat16)
        print("Text-to-Image pipeline loaded successfully!")
    return t2i_pipe


def load_edit_pipeline():
    """加载图像编辑模型"""
    global edit_pipe
    if edit_pipe is None:
        print("Loading Image Edit pipeline...")
        text_processor = AutoProcessor.from_pretrained(EDIT_CHECKPOINT_DIR, subfolder='tokenizer')
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            EDIT_CHECKPOINT_DIR,
            subfolder='transformer',
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
        
        edit_pipe = LongCatImageEditPipeline.from_pretrained(
            EDIT_CHECKPOINT_DIR,
            transformer=transformer,
            text_processor=text_processor
        )
        edit_pipe.to(device, torch.bfloat16)
        print("Image Edit pipeline loaded successfully!")
    return edit_pipe


def text_to_image(
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    enable_cfg_renorm: bool,
    enable_prompt_rewrite: bool
):
    """文生图推理"""
    if not prompt.strip():
        return None, "请输入提示词"
    
    try:
        pipe = load_t2i_pipeline()
        generator = torch.Generator("cpu").manual_seed(seed)
        
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
            enable_cfg_renorm=enable_cfg_renorm,
            enable_prompt_rewrite=enable_prompt_rewrite
        ).images[0]
        
        return image, "生成成功！"
    except Exception as e:
        return None, f"生成失败: {str(e)}"


def image_edit(
    input_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int
):
    """图像编辑推理"""
    if input_image is None:
        return None, "请上传图片"
    if not prompt.strip():
        return None, "请输入编辑指令"
    
    try:
        pipe = load_edit_pipeline()
        generator = torch.Generator("cpu").manual_seed(seed)
        
        # 确保图像是 RGB 模式
        input_image = input_image.convert('RGB')
        
        image = pipe(
            input_image,
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator
        ).images[0]
        
        return image, "编辑成功！"
    except Exception as e:
        return None, f"编辑失败: {str(e)}"


def create_ui():
    """创建 Gradio 界面"""
    with gr.Blocks(title="LongCat-Image") as demo:
        gr.Markdown(
            """
            # 🐱 LongCat-Image
            基于 LongCat 的图像生成与编辑系统
            """
        )
        
        with gr.Tabs():
            # 文生图标签页
            with gr.TabItem("🎨 文生图 (Text-to-Image)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2i_prompt = gr.Textbox(
                            label="提示词 (Prompt)",
                            placeholder="请输入图像描述...",
                            lines=4,
                            value="一个年轻的亚裔女性，身穿黄色针织衫，搭配白色项链。她的双手放在膝盖上，表情恬静。背景是一堵粗糙的砖墙，午后的阳光温暖地洒在她身上，营造出一种宁静而温馨的氛围。"
                        )
                        
                        gr.Examples(
                            examples=[
                                ["一个年轻的亚裔女性，身穿黄色针织衫，搭配白色项链。她的双手放在膝盖上，表情恬静。背景是一堵粗糙的砖墙，午后的阳光温暖地洒在她身上，营造出一种宁静而温馨的氛围。"],
                                ["一张精美的中国风海报，上面写着「龙腾四海」四个金色大字，字体苍劲有力，背景是云雾缭绕的山水画，配以祥云和腾飞的金龙，整体风格古典大气。"],
                                ["一本打开的古籍，书页上用毛笔写着「天道酬勤」四个字，旁边放着一支毛笔和砚台，窗外透进柔和的阳光，营造出书香门第的氛围。"],
                                ["一家中式茶馆的招牌，上面写着「清心茶舍」，招牌采用木质材料，字体是古朴的楷书，周围装饰着竹叶和茶叶图案。"],
                            ],
                            inputs=[t2i_prompt],
                            label="提示词示例 (点击使用)"
                        )
                        
                        t2i_negative_prompt = gr.Textbox(
                            label="负面提示词 (Negative Prompt)",
                            placeholder="不想出现的元素...",
                            lines=2,
                            value=""
                        )
                        
                        with gr.Row():
                            t2i_height = gr.Slider(
                                minimum=256,
                                maximum=2048,
                                value=768,
                                step=64,
                                label="高度 (Height)"
                            )
                            t2i_width = gr.Slider(
                                minimum=256,
                                maximum=2048,
                                value=1344,
                                step=64,
                                label="宽度 (Width)"
                            )
                        
                        with gr.Row():
                            t2i_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=4.5,
                                step=0.5,
                                label="引导系数 (Guidance Scale)"
                            )
                            t2i_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=1,
                                label="推理步数 (Steps)"
                            )
                        
                        t2i_seed = gr.Number(
                            label="随机种子 (Seed)",
                            value=43,
                            precision=0
                        )
                        
                        with gr.Row():
                            t2i_cfg_renorm = gr.Checkbox(
                                label="启用 CFG Renorm",
                                value=True
                            )
                            t2i_prompt_rewrite = gr.Checkbox(
                                label="启用 Prompt Rewrite",
                                value=True
                            )
                        
                        with gr.Row():
                            t2i_btn = gr.Button("🚀 生成图像", variant="primary")
                            t2i_send_to_edit_btn = gr.Button("📤 发送到编辑", variant="secondary")
                    
                    with gr.Column(scale=1):
                        t2i_output = gr.Image(label="生成结果", type="pil")
                        t2i_status = gr.Textbox(label="状态", interactive=False)
                
                t2i_btn.click(
                    fn=text_to_image,
                    inputs=[
                        t2i_prompt,
                        t2i_negative_prompt,
                        t2i_height,
                        t2i_width,
                        t2i_guidance_scale,
                        t2i_steps,
                        t2i_seed,
                        t2i_cfg_renorm,
                        t2i_prompt_rewrite
                    ],
                    outputs=[t2i_output, t2i_status]
                )
            
            # 图像编辑标签页
            with gr.TabItem("✏️ 图像编辑 (Image Edit)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        edit_input_image = gr.Image(
                            label="上传图片",
                            type="pil"
                        )
                        edit_prompt = gr.Textbox(
                            label="编辑指令 (Edit Prompt)",
                            placeholder="请输入编辑指令，例如：将猫变成狗",
                            lines=3,
                            value="把衣服换成红色的"
                        )
                        
                        gr.Examples(
                            examples=[
                                ["把衣服换成红色的"],
                                ["把“清心茶舍”改成“悦来客栈”"],
                                ["把背景换成海边"],
                                ["给人物戴上眼镜"],
                            ],
                            inputs=[edit_prompt],
                            label="编辑指令示例 (点击使用)"
                        )
                        
                        edit_negative_prompt = gr.Textbox(
                            label="负面提示词 (Negative Prompt)",
                            placeholder="不想出现的元素...",
                            lines=2,
                            value=""
                        )
                        
                        with gr.Row():
                            edit_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=4.5,
                                step=0.5,
                                label="引导系数 (Guidance Scale)"
                            )
                            edit_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=1,
                                label="推理步数 (Steps)"
                            )
                        
                        edit_seed = gr.Number(
                            label="随机种子 (Seed)",
                            value=43,
                            precision=0
                        )
                        
                        edit_btn = gr.Button("🎯 编辑图像", variant="primary")
                    
                    with gr.Column(scale=1):
                        edit_output = gr.Image(label="编辑结果", type="pil")
                        edit_status = gr.Textbox(label="状态", interactive=False)
                
                edit_btn.click(
                    fn=image_edit,
                    inputs=[
                        edit_input_image,
                        edit_prompt,
                        edit_negative_prompt,
                        edit_guidance_scale,
                        edit_steps,
                        edit_seed
                    ],
                    outputs=[edit_output, edit_status]
                )
        
        # 发送到编辑按钮的事件：将生成的图片发送到编辑页面
        def send_to_edit(image):
            if image is None:
                return gr.update(), "请先生成图片"
            return image, "图片已发送到编辑页面，请切换到「图像编辑」标签页"
        
        t2i_send_to_edit_btn.click(
            fn=send_to_edit,
            inputs=[t2i_output],
            outputs=[edit_input_image, t2i_status]
        )
        
        gr.Markdown(
            """
            ---
            ### 使用说明
            - **文生图**: 输入文字描述，AI 将根据描述生成对应的图像
            - **图像编辑**: 上传一张图片，输入编辑指令，AI 将对图片进行相应的修改
            - **随机种子**: 使用相同的种子和参数可以复现相同的结果
            - **引导系数**: 值越高，生成结果越贴近提示词，但可能降低多样性
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
