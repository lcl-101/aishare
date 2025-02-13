#使用 gradio 生成一个页面
#用户可以上传一个图像文件，上传之后，将由 RMBG 处理，将结果返回给用户

import os
import gradio as gr
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

# 模型初始化
torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-2.0", trust_remote_code=True
)
birefnet.to("cuda")
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
# 定义输出文件夹，如果不存在则创建
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义去除背景的处理函数
def fnrmbg(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    image = process(im)    
    image_path = os.path.join(output_folder, "no_bg_image.png")
    image.save(image_path)
    return image_path  # 只返回图像文件路径

#定义数据处理函数
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

#控件定义
demo = gr.Interface(
    fn=fnrmbg, 
    inputs = gr.Image(type="filepath", label="Input Image"),
    outputs = gr.Image(type="filepath", label="Output Image"),
    title='RMBG 测试',
)

if __name__ == "__main__":
    demo.launch()
