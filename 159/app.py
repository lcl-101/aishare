import sys
sys.path.append("../")

import gradio as gr
import os 
from PIL import Image
import torch
from diffusers.utils import check_min_version
from tools.download_util import load_file_from_url
from tools.painter import mask_painter
import argparse
from objectclear.pipelines import ObjectClearPipeline
import numpy as np
import torchvision.transforms.functional as TF
from scipy.ndimage import convolve, zoom
import cv2
import time

from tools.interact_tools import SamControler
from tools.misc import get_device
import json

check_min_version("0.30.2")


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=7860, help="only useful when running gradio applications")  
    args = parser.parse_args()
    
    if not args.device:
        args.device = str(get_device())

    return args 


def pad_to_multiple(image: np.ndarray, multiple: int = 8):
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if image.ndim == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), mode='reflect')
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded, h, w

def crop_to_original(image: np.ndarray, h: int, w: int):
    return image[:h, :w]

def wavelet_blur_np(image: np.ndarray, radius: int):
    kernel = np.array([
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625]
    ], dtype=np.float32)

    blurred = np.empty_like(image)
    for c in range(image.shape[0]):
        blurred_c = convolve(image[c], kernel, mode='nearest')
        if radius > 1:
            blurred_c = zoom(zoom(blurred_c, 1 / radius, order=1), radius, order=1)
        blurred[c] = blurred_c
    return blurred

def wavelet_decomposition_np(image: np.ndarray, levels=5):
    high_freq = np.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur_np(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq

def wavelet_reconstruction_np(content_feat: np.ndarray, style_feat: np.ndarray):
    content_high, _ = wavelet_decomposition_np(content_feat)
    _, style_low = wavelet_decomposition_np(style_feat)
    return content_high + style_low

def wavelet_color_fix_np(fused: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fused_np = fused.astype(np.float32) / 255.0
    mask_np = mask.astype(np.float32) / 255.0

    fused_np = fused_np.transpose(2, 0, 1)
    mask_np = mask_np.transpose(2, 0, 1)

    result_np = wavelet_reconstruction_np(fused_np, mask_np)

    result_np = result_np.transpose(1, 2, 0)
    result_np = np.clip(result_np * 255.0, 0, 255).astype(np.uint8)

    return result_np

def fuse_with_wavelet(ori: np.ndarray, removed: np.ndarray, attn_map: np.ndarray, multiple: int = 8):
    H, W = ori.shape[:2]
    attn_map = attn_map.astype(np.float32)
    _, attn_map = cv2.threshold(attn_map, 128, 255, cv2.THRESH_BINARY)
    am = attn_map.astype(np.float32)
    am = am/255.0
    am_up = cv2.resize(am, (W, H), interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    am_d = cv2.dilate(am_up, kernel, iterations=1)
    am_d = cv2.GaussianBlur(am_d.astype(np.float32), (9,9), sigmaX=2)

    am_merged = np.maximum(am_up, am_d)
    am_merged = np.clip(am_merged, 0, 1)

    attn_up_3c = np.stack([am_merged]*3, axis=-1)
    attn_up_ori_3c = np.stack([am_up]*3, axis=-1)

    ori_out = ori * (1 - attn_up_ori_3c)
    rem_out = removed * (1 - attn_up_ori_3c)

    ori_pad, h0, w0 = pad_to_multiple(ori_out, multiple)
    rem_pad, _, _   = pad_to_multiple(rem_out, multiple)

    wave_rgb = wavelet_color_fix_np(ori_pad, rem_pad)
    wave = crop_to_original(wave_rgb, h0, w0)
    # fusion
    fused = (wave * (1 - attn_up_3c) + removed * attn_up_3c).astype(np.uint8)
    return fused


def resize_by_short_side(image, target_short=512, resample=Image.BICUBIC):
    w, h = image.size
    if w < h:
        new_w = target_short
        new_h = int(h * target_short / w)
        new_h = (new_h + 15) // 16 * 16 
    else:
        new_h = target_short
        new_w = int(w * target_short / h)
        new_w = (new_w + 15) // 16 * 16
    return image.resize((new_w, new_h), resample=resample)

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

# use sam to get the mask
def sam_refine(image_state, point_prompt, click_state, evt:gr.SelectData):
    if point_prompt == "正向":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(image_state["origin_image"])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=image_state["origin_image"], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    image_state["mask"] = mask
    image_state["logit"] = logit
    image_state["painted_image"] = painted_image

    return painted_image, image_state, click_state


def add_multi_mask(image_state, interactive_state, mask_dropdown):
    mask = image_state["mask"]
    interactive_state["masks"].append(mask)
    interactive_state["mask_names"].append("mask_{:03d}".format(len(interactive_state["masks"])))
    mask_dropdown.append("mask_{:03d}".format(len(interactive_state["masks"])))
    select_frame = show_mask(image_state, interactive_state, mask_dropdown)

    return interactive_state, gr.update(choices=interactive_state["mask_names"], value=mask_dropdown), select_frame, [[],[]]

def clear_click(image_state, click_state):
    click_state = [[],[]]
    input_image = image_state["origin_image"]
    return input_image, click_state

def remove_multi_mask(interactive_state, click_state, image_state):
    interactive_state["mask_names"]= []
    interactive_state["masks"] = []
    click_state = [[],[]]
    input_image = image_state["origin_image"]

    return interactive_state, gr.update(choices=[],value=[]), input_image, click_state

def show_mask(image_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    if image_state["origin_image"] is not None:
        select_frame = image_state["origin_image"]
        for i in range(len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            mask = interactive_state["masks"][mask_number]
            select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
        
        return select_frame

def upload_and_reset(image_input, interactive_state):
    click_state = [[], []]

    interactive_state["mask_names"]= []
    interactive_state["masks"] = []

    image_state, image_info, image_input = update_image_state_on_upload(image_input)

    return (
        image_state,
        image_info,
        image_input,
        interactive_state,
        click_state,
        gr.update(choices=[], value=[]),
    )

def update_image_state_on_upload(image_input):
    frame = image_input 

    image_size = (frame.size[1], frame.size[0])

    frame_np = np.array(frame)

    image_state = {
        "origin_image": frame_np,
        "painted_image": frame_np.copy(),
        "mask": np.zeros((image_size[0], image_size[1]), np.uint8),
        "logit": None,
    }

    image_info = f"图片名称: uploaded.png,\n图片尺寸: {image_size}"
    
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(frame_np)

    return image_state, image_info, image_input
        

# SAM generator
class MaskGenerator():
    def __init__(self, sam_checkpoint, args):
        self.args = args
        self.samcontroler = SamControler(sam_checkpoint, args.sam_model_type, args.device)

    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image
    

# args, defined in track_anything.py
args = parse_augment()
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type])
# initialize sams
model = MaskGenerator(sam_checkpoint, args)

# Build pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
    "jixin0101/ObjectClear",
    torch_dtype=torch.float16,
    variant='fp16',
    apply_attention_guided_fusion=True,
    cache_dir=None,
)

pipe.to(device)

def process(image_state, interactive_state, mask_dropdown, guidance_scale, seed, num_inference_steps, strength         
            ):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image_np = image_state["origin_image"]
    image = Image.fromarray(image_np)
    if interactive_state["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        image_state["mask"]= template_mask
    else:      
        template_mask = image_state["mask"]
    mask = Image.fromarray((template_mask).astype(np.uint8) * 255)
    image_or = image.copy()
    
    image = image.convert("RGB")
    mask = mask.convert("RGB")
    
    image = resize_by_short_side(image, 512, resample=Image.BICUBIC)
    mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)
    
    w, h = image.size
    
    result = pipe(
        prompt="remove the instance of object",
        image=image,
        mask_image=mask,
        generator=generator,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=guidance_scale,
        height=h,
        width=w,
    )
    fused_img_pil = result.images[0]

    return fused_img_pil.resize((image_or.size[:2])), (image.resize((image_or.size[:2])), fused_img_pil.resize((image_or.size[:2])))

import base64
with open("./Logo.png", "rb") as f:
    img_bytes = f.read()
img_b64 = base64.b64encode(img_bytes).decode()

html_img = f'''
<div style="display:flex; justify-content:center; align-items:center; width:100%;">
    <img src="data:image/png;base64,{img_b64}" style="border:none; width:200px; height:auto;"/>
</div>
'''

tutorial_url = "https://github.com/zjx0101/ObjectClear/releases/download/media/tutorial.mp4"
assets_path = './'
load_file_from_url(tutorial_url, assets_path)

description = r"""
<b>官方 Gradio 演示</b> for <a href='https://github.com/zjx0101/ObjectClear' target='_blank'><b>ObjectClear: 通过物体效应注意力实现完整物体移除</b></a>.<br>
🔥 ObjectClear 是一个物体移除模型，可以利用物体效应注意力机制联合消除目标物体及其相关效应，同时保持背景一致性。<br>
🖼️ 试试上传您的图片，通过几次点击指定目标遮罩，并获得物体移除结果！<br>

*注意：所有输入图像在推理期间会临时调整大小（短边 = 512 像素）以匹配训练分辨率。最终输出会恢复到原始分辨率。<br>*
"""

article = r"""<h3>
<b>如果 ObjectClear 对您有帮助，请帮忙给 <a href='https://github.com/zjx0101/ObjectClear' target='_blank'>Github 仓库</a> 点个星。谢谢！</b></h3>
<hr>

📑 **引用**
<br>
如果我们的工作对您的研究有用，请考虑引用：
```bibtex
@InProceedings{zhao2025ObjectClear,
    title     = {{ObjectClear}: Complete Object Removal via Object-Effect Attention},
    author    = {Zhao, Jixin and Zhou, Shangchen and Wang, Zhouxia and Yang, Peiqing and Loy, Chen Change},
    booktitle = {arXiv preprint arXiv:2505.22636},
    year      = {2025}
    }
```
📧 **联系方式**
<br>
如有任何问题，请随时联系我：<b>jixinzhao0101@gmail.com</b>。
<br>
👏 **致谢**
<br>
此演示改编自 [MatAnyone](https://github.com/pq-yang/MatAnyone)，并利用了 [Segment Anything](https://github.com/facebookresearch/segment-anything) 的分割功能。感谢他们的出色工作！
"""

custom_css = """
#input-image {
    aspect-ratio: 1 / 1;
    width: 100%;
    max-width: 100%;
    height: auto;
    display: flex;
    align-items: center;
    justify-content: center;
}

#input-image img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
}

#main-columns {
    gap: 60px; 
}

#main-columns > .gr-column {
    flex: 1; 
}

#compare-image {
    width: 100%;
    aspect-ratio: 1 / 1; 
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0;
    padding: 0;
    max-width: 100%;
    box-sizing: border-box;
}

#compare-image svg.svelte-zyxd38 {
    position: absolute !important; 
    top: 50% !important;           
    left: 50% !important;          
    transform: translate(-50%, -50%) !important; 
}

#compare-image .icon.svelte-1oiin9d {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

#compare-image {
    position: relative;
    overflow: hidden;
}

.new_button {background-color: #171717 !important; color: #ffffff !important; border: none !important;}
.new_button:hover {background-color: #4b4b4b !important;}

#start-button {
    background: linear-gradient(135deg, #2575fc 0%, #6a11cb 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 12px;
    cursor: pointer;
    box-shadow: 0 0 12px rgba(100, 100, 255, 0.7);
    transition: all 0.3s ease;
}
#start-button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(100, 100, 255, 1);
}

<style>
.button-wrapper {
    width: 30%;
    text-align: center; 
}
.wide-button {
    width: 83% !important;
    background-color: black !important;
    color: white !important;
    border: none !important;
    padding: 8px 0 !important;
    font-size: 16px !important;
    display: inline-block;
    margin: 30px 0px 0px 50px ;
}
.wide-button:hover {
    background-color: #656262 !important;
}
</style>
"""


with gr.Blocks(css=custom_css) as demo:
    gr.HTML(html_img)
    gr.Markdown(description)
    with gr.Group(elem_classes="gr-monochrome-group", visible=True):
        with gr.Row():
            with gr.Accordion('SAM 设置 (点击展开)', open=False):
                with gr.Row():
                    point_prompt = gr.Radio(
                        choices=["正向", "负向"],
                        value="正向",
                        label="点提示",
                        info="点击添加正向或负向点来定义目标遮罩",
                        interactive=True,
                        min_width=100,
                        scale=1)
                    mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="遮罩选择", info="选择在步骤2中添加的1个或多个遮罩")
                    
    with gr.Row(elem_id="main-columns"):
        with gr.Column():
            
            click_state = gr.State([[],[]])

            interactive_state = gr.State(
                {
                    "mask_names": [],
                    "masks": []
                }
            )

            image_state = gr.State(
                {
                "origin_image": None,
                "painted_image": None,
                "mask": None,
                "logit": None
                }
            )
            
            image_info = gr.Textbox(label="图片信息", visible=False)
            input_image = gr.Image(
                label='输入图片',
                type='pil',
                sources=["upload"],
                image_mode='RGB',
                interactive=True,
                elem_id="input-image"
            )
            
            with gr.Row(equal_height=True, elem_classes="mask_button_group"):
                clear_button_click = gr.Button(value="清除点击",elem_classes="new_button", min_width=100)
                add_mask_button = gr.Button(value="添加遮罩", elem_classes="new_button", min_width=100)
                remove_mask_button = gr.Button(value="删除遮罩", elem_classes="new_button", min_width=100)
                
            submit_button_component = gr.Button(
                value='开始 ObjectClear', elem_id="start-button"
            )
            
            with gr.Accordion('ObjectClear 设置', open=True):
                strength = gr.Radio(
                    choices=[0.99, 1.0],
                    value=0.99,
                    label="强度",
                    info="0.99 更好地保留背景和颜色；如果物体/阴影未完全移除，请使用 1.0（默认：0.99）"
                )
                
                guidance_scale = gr.Slider(
                    minimum=1, maximum=10, step=0.5, value=2.5,
                    label="引导比例",
                    info="更高 = 更强的移除效果；更低 = 更好的背景保留（默认：2.5）"
                )
                
                seed = gr.Slider(
                    minimum=0, maximum=1000000, step=1, value=300000,
                    label="随机种子",
                    info="不同的种子可能导致明显不同的物体移除结果（默认：300000）"
                )
                
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=40, step=1, value=20,
                    label="推理步数",
                    info="更高的值可能提高质量但需要更长时间（默认：20）"
                )
            
            
        with gr.Column():
            output_image_component = gr.Image(
                type='pil', image_mode='RGB', label='输出结果', format="png", elem_id="input-image")
            
            output_compare_image_component = gr.ImageSlider(
                label="对比图", 
                type="pil", 
                format='png', 
                elem_id="compare-image"
            )

        input_image.upload(
            fn=upload_and_reset,
            inputs=[input_image, interactive_state],
            outputs=[
                image_state,
                image_info,
                input_image,
                interactive_state,
                click_state,
                mask_dropdown,
            ]
        )

        # click select image to get mask using sam
        input_image.select(
            fn=sam_refine,
            inputs=[image_state, point_prompt, click_state],
            outputs=[input_image, image_state, click_state]
        )
        
        # add different mask
        add_mask_button.click(
            fn=add_multi_mask,
            inputs=[image_state, interactive_state, mask_dropdown],
            outputs=[interactive_state, mask_dropdown, input_image, click_state]
        )

        remove_mask_button.click(
            fn=remove_multi_mask,
            inputs=[interactive_state, click_state, image_state],
            outputs=[interactive_state, mask_dropdown, input_image, click_state]
        )
    
        # points clear
        clear_button_click.click(
            fn = clear_click,
            inputs = [image_state, click_state,],
            outputs = [input_image, click_state],
        )
    
    submit_button_component.click(
        fn=process,
        inputs=[
            image_state, 
            interactive_state,
            mask_dropdown,
            guidance_scale,
            seed,
            num_inference_steps,
            strength
        ],
        outputs=[
            output_image_component, output_compare_image_component
        ]
    )
    
    with gr.Accordion("📕 视频教程 (点击展开)", open=False, elem_classes="custom-bg"):
        with gr.Row():
            gr.Video(value="./tutorial.mp4", elem_classes="video")

    gr.Markdown("---")
    gr.Markdown("## 示例")

    example_images = [
        os.path.join(os.path.dirname(__file__), "examples", f"test{i}.png") 
        for i in range(10)
    ]
    
    examples_data = [
        [example_images[i], None] for i in range(len(example_images))
    ]

    examples = gr.Examples(
        examples=examples_data,
        inputs=[input_image, interactive_state],
        outputs=[image_state, image_info, input_image,
                interactive_state, click_state, mask_dropdown],
        fn=upload_and_reset,
        run_on_click=True,
        cache_examples=False,
        label="点击下方加载示例图像"
    )
    
    gr.Markdown(article)


demo.launch(debug=True, show_error=True, server_name="0.0.0.0",server_port=args.port)