import os
import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "checkpoints/aya-vision-8b"
device = "cuda:0"  # adjust if necessary

# Read sample images from the samples directory
sample_images = [os.path.join("samples", f) for f in os.listdir("samples") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
# Examples for demo1: each example is a list with one image and one prompt
demo1_examples = [[img, "图片中的文字写了什么？"] for img in sample_images]
# Examples for demo3: use first two images if available, otherwise empty list, with a default prompt
demo3_examples = []
if len(sample_images) >= 2:
    demo3_examples = [[sample_images[2], sample_images[3], "These images depict two different landmarks. Can you identify them?"]]

# Initialize processor and model (using fast processor for demo1)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map=device, torch_dtype=torch.float16
)

def run_demo1(image, prompt_text):
    # Build conversation for a single image test with custom prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages, padding=True, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", device=device
    ).to(model.device)
    gen_tokens = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.3
    )
    output_text = processor.tokenizer.decode(
        gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return output_text

def run_demo3(image1, image2, prompt_text):
    # Build conversation for two uploaded images test with custom prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages, padding=True, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", device=device
    ).to(model.device)
    gen_tokens = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.3
    )
    output_text = processor.tokenizer.decode(
        gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return output_text

# Create examples with specific prompts for each image
demo1_examples = []
for img in sample_images:
    # Set specific prompts for different images
    if "2.jpg" in img:
        prompt = "Which monument is shown in this picture?"
    elif "5.jpg" in img:
        prompt = "请详细描述一下图中所表达的意境。"
    elif "6.jpg" in img:
        prompt = "请尝试找出图中左侧汽车与右侧汽车的三个不同之处"
    else:
        prompt = "图片中的文字写了什么？"
    demo1_examples.append([img, prompt])

# Define Gradio Interface for demo1
demo1_interface = gr.Interface(
    fn=run_demo1, 
    inputs=[
        gr.Image(type="pil", label="上传图片"), 
        gr.Textbox(label="提示词", value="图片中的文字写了什么？")
    ], 
    outputs="text", 
    title="Demo1: 单张图片测试",
    examples=demo1_examples
)

demo3_interface = gr.Interface(
    fn=run_demo3, 
    inputs=[
        gr.Image(type="pil", label="上传图片1"), 
        gr.Image(type="pil", label="上传图片2"),
        gr.Textbox(label="提示词", value="These images depict two different landmarks. Can you identify them?")
    ], 
    outputs="text", 
    title="Demo3: 双图片测试",
    examples=demo3_examples
)

# Create a tabbed interface with two tabs
app = gr.TabbedInterface(
    [demo1_interface, demo3_interface],
    tab_names=["单图", "多图"]
)

app.launch(server_name="0.0.0.0")
