import os
import time
import random
import gradio as gr
import torch
from PIL import Image


"""Resolve local model root before importing pipeline.
We try common layouts and validate they contain expected subfolders.
"""
HERE = os.path.dirname(__file__)

def _is_valid_root(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    # Minimal structure check
    expected = [
        os.path.join(path, "vae", "vae_2_1"),
        os.path.join(path, "dit"),
        os.path.join(path, "text_encoder"),
    ]
    return all(os.path.exists(p) for p in expected)

env_root = os.environ.get("HUNYUANIMAGE_V2_1_MODEL_ROOT", "")
candidate_paths = [
    env_root,
    os.path.abspath(os.path.join(HERE, "ckpts", "HunyuanImage-2.1")),
    os.path.abspath(os.path.join(HERE, "ckpts", "ckpts")),
    os.path.abspath(os.path.join(HERE, "ckpts")),
    os.path.abspath(os.path.join(HERE, "HunyuanImage-2.1")),
]
for cand in [p for p in candidate_paths if p]:
    if _is_valid_root(cand):
        os.environ["HUNYUANIMAGE_V2_1_MODEL_ROOT"] = cand
        break

# Prefer offline mode to prevent any remote fetch attempts when local files exist
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class HYWebUI:
    def __init__(self):
        self.pipe = None
        self.device = get_device()
        self.root = os.environ.get("HUNYUANIMAGE_V2_1_MODEL_ROOT", "")

    def load_pipeline(self, model_name: str, offloading: bool):
        if self.pipe is not None:
            return self.pipe
        self.pipe = HunyuanImagePipeline.from_pretrained(model_name=model_name, torch_dtype='bf16').to(self.device)
        # Toggle offloading inversely: if offloading False, keep everything on GPU
        self.pipe.update_config(
            enable_dit_offloading=offloading,
            enable_reprompt_model_offloading=offloading,
            enable_refiner_offloading=offloading,
        )
        return self.pipe

    def generate(self, prompt: str, negative_prompt: str, model_name: str, width: int, height: int,
                 use_reprompt: bool, use_refiner: bool, steps: int, guidance: float, shift: int,
                 seed: int | None, offloading: bool):
        t0 = time.time()
        pipe = self.load_pipeline(model_name, offloading)

        # Check optional components based on local files
        if use_reprompt and not os.path.isdir(os.path.join(self.root, "reprompt")):
            use_reprompt = False
        if use_refiner and not (
            os.path.isdir(os.path.join(self.root, "vae", "vae_refiner")) and
            os.path.isfile(os.path.join(self.root, "dit", "hunyuanimage-refiner.safetensors"))
        ):
            use_refiner = False

        # Preload optional models to GPU when offloading is False
        if not offloading:
            if use_reprompt:
                _ = pipe.reprompt_model
                pipe.reprompt_model.to(pipe.execution_device)
            if use_refiner:
                _ = pipe.refiner_pipeline
                pipe.refiner_pipeline.to(pipe.execution_device)

        # Seed handling
        if seed is None or seed < 0:
            seed = random.randint(0, 2**31 - 1)

        image: Image.Image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            use_reprompt=use_reprompt,
            use_refiner=use_refiner,
            num_inference_steps=steps,
            guidance_scale=guidance,
            shift=shift,
            seed=seed,
        )

        t1 = time.time()
        info = f"Model: {model_name} | Steps: {steps} | CFG: {guidance} | Shift: {shift} | Size: {width}x{height} | Seed: {seed} | Time: {t1 - t0:.1f}s"
        return image, info


app = HYWebUI()


def ui():
    with gr.Blocks(title="HunyuanImage 2.1 WebUI") as demo:
        gr.Markdown("## HunyuanImage 2.1 WebUI")
        RES_PRESETS = {
            "2560x1536 (16:9)": (2560, 1536),
            "2304x1792 (4:3)": (2304, 1792),
            "2048x2048 (1:1)": (2048, 2048),
            "1792x2304 (3:4)": (1792, 2304),
            "1536x2560 (9:16)": (1536, 2560),
        }
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Describe the image you want...")
                negative = gr.Textbox(label="Negative Prompt", lines=3)
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=["hunyuanimage-v2.1", "hunyuanimage-v2.1-distilled"],
                        value="hunyuanimage-v2.1",
                        label="Model"
                    )
                    offloading = gr.Checkbox(value=False, label="Enable CPU Offloading (saves VRAM)")
                resolution = gr.Dropdown(
                    choices=list(RES_PRESETS.keys()),
                    value="2048x2048 (1:1)",
                    label="Resolution"
                )
                with gr.Row():
                    steps = gr.Slider(4, 60, value=50, step=1, label="Steps (8 for distilled)")
                    guidance = gr.Slider(1.0, 9.0, value=3.5, step=0.1, label="Guidance Scale")
                with gr.Row():
                    shift = gr.Slider(1, 8, value=5, step=1, label="Shift")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1=Random)")
                with gr.Row():
                    use_reprompt = gr.Checkbox(value=True, label="Use Reprompt")
                    use_refiner = gr.Checkbox(value=True, label="Use Refiner")
                gr.Examples(
                    examples=[
                        [
                            """一幅由四个画格组成的卡通漫画，以2x2的网格形式排列，讲述了一只变色龙的难题。 左上角：一只卡通风格的变色龙趴在一片宽大的、呈翠绿色的植物叶子上。它的皮肤是与叶片完全相同的绿色，并带有浅绿色的纹理细节，实现了完美的伪装，几乎与背景融为一体。变色龙的大眼睛好奇地转动着，身体姿态放松，场景背景是模糊的绿色丛林。 右上角：同一只变色龙正从叶子边缘爬到一根粗糙的、呈深棕色的树枝上。它的身体颜色已经完全转变为与树枝一致的深棕色，皮肤表面模仿出树皮的纹理。它的头部微微抬起，嘴角上扬，眼神中流露出自豪和得意的神情。 左下角：这只变色龙自信地走到一片铺在草地上的野餐布前。它的一只前爪已经踏上了野餐布，野餐布是由红白相间的方格图案构成。此时变色龙的身体仍然保持着棕色，它正准备完全走上这块图案复杂的布料，表情显得充满期待。 右下角：变色龙完全站在了红白格子野餐布的中央。它的颜色系统出现了故障，身体表面在多种颜色和图案之间混乱地快速闪烁，包括霓虹粉的斑点、电光蓝的条纹和像素化的色块，完全无法匹配背景的格子图案。它的眼睛睁得滚圆，嘴巴大张呈惊恐的O形，身体周围出现了表示慌乱的动态线条和汗珠。 这组图像整体呈现出线条清晰、色彩鲜明的四格漫画作品风格。""",
                            "",
                            "hunyuanimage-v2.1",
                            "2048x2048 (1:1)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            """创意交通信号灯，灯罩内从上到下分别是：顶部一个方形红色机器人（双臂交叉，表情严肃）、中间一个圆形黄色机器人（旋转天线，疑惑表情）、底部一个三角形绿色机器人（双臂上举，欢呼表情），取代传统信号灯。信号灯带有铆钉装饰的深灰色金属灯杆和结构。背景是清澈的蓝天和蓬松的白云。整体场景明亮欢快，具有卡通、俏皮的风格，玩具机器人风格，金属质感，关节分明。""",
                            "",
                            "hunyuanimage-v2.1",
                            "2048x2048 (1:1)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            """星空下，一个充满未来感的泳池，映照着深邃的宇宙。泳池表面闪耀着星云、旋转的星系和闪烁的星光。青色、洋红色和紫色的霓虹灯照亮了整个区域，营造出令人着迷的赛博朋克氛围。泳池上方，紫色和粉色的文字“Prompt Enhancer”漂浮在半空中，周围环绕着柔和的光晕，在水面上投射出文字“Prompt Enhancer”反射的光芒。背景是带有空灵光环和宇宙尘埃的遥远行星，增强了超现实梦幻的氛围。场景将时尚的现代建筑与超凡脱俗的科技设计相结合，营造出一种奇妙的感觉和未来主义的优雅。""",
                            "",
                            "hunyuanimage-v2.1",
                            "2560x1536 (16:9)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            "一片翠绿的森林中，树木和藤蔓构成了“林”字，树枝弯曲成字形，鸟儿栖息在树枝上，阳光透过树叶洒下，整个画面自然且充满生气。",
                            "",
                            "hunyuanimage-v2.1",
                            "2048x2048 (1:1)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            "A close-up shot focuses on a young girl with vibrant, curly hair and a gentle expression. Her face is characterized by a light dusting of sun-kissed freckles across her nose and cheeks. She wears a dress with a colorful floral pattern and holds a freshly-picked bouquet of wildflowers, including daisies and lavender, against a softly blurred background. The image presents a realistic photography style.",
                            "",
                            "hunyuanimage-v2.1",
                            "1792x2304 (3:4)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            "在酒馆外面，一个卖报的小男孩带着报童帽，倚靠着一根大理石质地的罗马柱，罗马柱靠右，他穿着吊带裤，蹲在地上，背靠着柱子，面对着镜头，侧着耳朵，专注地倾听酒馆内传来的钢琴声。酒馆内部，一个少女背对着镜头，坐在老式木质钢琴前。扎着两条麻花辫，侧颜微微可见，高挺的鼻梁和微卷的发丝清晰可见。她身穿一件米白色的蕾丝连衣裙，正在弹奏钢琴。莫奈的印象派画面，营造出一种夏日浪漫宁静的氛围。",
                            "",
                            "hunyuanimage-v2.1",
                            "2304x1792 (4:3)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            "一只Q版拟人的小企鹅，戴着红色针织围巾和贝雷帽，手里拿着单反相机，像街头摄影师一样站在人行道上，写实摄影风格",
                            "",
                            "hunyuanimage-v2.1",
                            "2048x2048 (1:1)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            "一幅超现实主义风格的悟空手办，人物漂浮在半空，身体部分化为液态银色金属，眼神锐利，手拿着金箍棒，背景是一条巨龙在云雾中盘旋，鳞片闪烁光芒，场景充满力量感，空中有闪电。",
                            "",
                            "hunyuanimage-v2.1",
                            "2048x2048 (1:1)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            "A wildlife poster design for the Serengeti plains features a central illustration of a chibi-style explorer riding a lion cub, set against a backdrop of rolling hills. At the top of the composition, the title \"Serengeti: Roar of Adventure\" is displayed in a large, whimsical font with decorative, swirling letters. The main scene depicts a wide-eyed chibi explorer, characterized by a large head and a small body, sitting atop a friendly lion cub. The explorer wears a green explorer's hat, a backpack, and holds onto the cub's mane, looking forward with a look of wonder. The lion cub, with a light brown mane and a smiling expression, strides forward, its body rendered in warm orange tones. In the background, the Serengeti plains are illustrated with rolling hills and savanna grass, all in shades of warm yellow and soft brown. Below the main illustration, the tagline \"Where Dreams Run Wild\" is written in a smaller, elegant script. The overall presentation is that of a poster design, combining a cute chibi illustration style with playful, whimsical typography.",
                            "",
                            "hunyuanimage-v2.1",
                            "1792x2304 (3:4)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                        [
                            "A vibrant winter wonderland poster for Lapland is presented, featuring a central illustration of a fluffy fox sledding down a snowy hill beneath an aurora borealis sky. In the foreground, a small, fluffy fox with bright orange fur is depicted joyfully riding a simple wooden sled. The fox wears a cozy, patterned scarf around its neck, and its tail curls happily as it glides over the snow. The background is dominated by a spectacular aurora borealis, with swirling curtains of light in shades of icy cyan and pink dancing across the night sky, interspersed with sparkling white snowflakes. Snow-covered mountains and pine trees are visible beneath the aurora. At the top of the poster, the text \"Lapland Magic\" is displayed in a large, playful font that resembles snowflakes. A smaller tagline, \"Find Your Frosty Adventure,\" is written in an elegant script below the main title. The overall color palette consists of icy cyan, pink, and sparkling white, creating a magical and inviting atmosphere. The presentation is that of a graphic design poster, combining illustration and typography.",
                            "",
                            "hunyuanimage-v2.1",
                            "1792x2304 (3:4)",
                            True,
                            True,
                            50,
                            3.5,
                            5,
                            -1,
                            False,
                        ],
                    ],
                    inputs=[
                        prompt,
                        negative,
                        model_name,
                        resolution,
                        use_reprompt,
                        use_refiner,
                        steps,
                        guidance,
                        shift,
                        seed,
                        offloading,
                    ],
                    label="Examples",
                    examples_per_page=10,
                )
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                image = gr.Image(label="Result", type="pil")
                info = gr.Textbox(label="Info", interactive=False)

        def _on_generate(prompt, negative, model_name, resolution_key, use_reprompt, use_refiner, steps, guidance, shift, seed, offloading):
            w, h = RES_PRESETS.get(resolution_key, (2048, 2048))
            return app.generate(prompt, negative, model_name, int(w), int(h), use_reprompt, use_refiner, int(steps), float(guidance), int(shift), int(seed), bool(offloading))

        btn.click(_on_generate, inputs=[prompt, negative, model_name, resolution, use_reprompt, use_refiner, steps, guidance, shift, seed, offloading], outputs=[image, info])

    return demo


if __name__ == "__main__":
    ui().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
