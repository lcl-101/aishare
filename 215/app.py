import os
import uuid
import argparse

import gradio as gr
import torch
from diffusers.utils import load_image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from dreamomni2.pipeline_dreamomni2 import DreamOmni2Pipeline
from utils.vprocess import process_vision_info, resizeinput


def parse_args():
    parser = argparse.ArgumentParser(description="启动 DreamOmni2 Gradio 应用程序。")
    parser.add_argument(
        "--vlm_path",
        type=str,
        default="checkpoints/DreamOmni2/vlm-model",
        help="Qwen2_5_VL 多模态语言模型目录路径。",
    )
    parser.add_argument(
        "--edit_lora_path",
        type=str,
        default="checkpoints/DreamOmni2/edit_lora",
        help="FLUX.1-Kontext 编辑 LoRA 权重目录路径。",
    )
    parser.add_argument(
        "--gen_lora_path",
        type=str,
        default="checkpoints/DreamOmni2/gen_lora",
        help="FLUX.1-Kontext 生成 LoRA 权重目录路径。",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="用于托管 Gradio Demo 的服务器 IP。",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="用于托管 Gradio Demo 的端口号。",
    )
    return parser.parse_args()


ARGS = parse_args()
vlm_path = ARGS.vlm_path
edit_lora_path = ARGS.edit_lora_path
gen_lora_path = ARGS.gen_lora_path
server_name = ARGS.server_name
server_port = ARGS.server_port
device = "cuda"


def extract_gen_content(text: str) -> str:
    return text[6:-7]


print(
    f"正在加载模型：vlm_path={vlm_path}, edit_lora_path={edit_lora_path}, gen_lora_path={gen_lora_path}"
)

pipe = DreamOmni2Pipeline.from_pretrained(
    "checkpoints/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
)
pipe.to(device)

AVAILABLE_ADAPTERS = set()

if edit_lora_path and os.path.exists(edit_lora_path):
    pipe.load_lora_weights(edit_lora_path, adapter_name="edit")
    AVAILABLE_ADAPTERS.add("edit")
else:
    print(f"警告：未在 {edit_lora_path} 找到编辑 LoRA 路径")

if gen_lora_path and os.path.exists(gen_lora_path):
    pipe.load_lora_weights(gen_lora_path, adapter_name="generation")
    AVAILABLE_ADAPTERS.add("generation")
else:
    print(f"警告：未在 {gen_lora_path} 找到生成 LoRA 路径")

vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_path,
    torch_dtype="bfloat16",
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(vlm_path)


def infer_vlm(input_img_path, input_instruction, prefix):
    if not vlm_model or not processor:
        raise gr.Error("未加载 VLM 模型，无法处理指令。")
    tp = []
    for path in input_img_path:
        tp.append({"type": "image", "image": path})
    tp.append({"type": "text", "text": input_instruction + prefix})
    messages = [{"role": "user", "content": tp}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=4096)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def find_closest_resolution(width, height, preferred_resolutions):
    input_ratio = width / height
    closest_resolution = min(
        preferred_resolutions,
        key=lambda res: abs((res[0] / res[1]) - input_ratio),
    )
    return closest_resolution


def perform_edit(input_img_paths, input_instruction, output_path):
    if "edit" not in AVAILABLE_ADAPTERS:
        raise gr.Error("未加载编辑适配器，请检查控制台警告。")
    pipe.set_adapters(["edit"], adapter_weights=[1])  # 切换到编辑适配器
    prefix = " It is editing task."
    source_imgs = []
    for path in input_img_paths:
        img = load_image(path)
        source_imgs.append(resizeinput(img))
    prompt = infer_vlm(input_img_paths, input_instruction, prefix)
    prompt = extract_gen_content(prompt)
    print(f"编辑模式生成提示：{prompt}")

    image = pipe(
        images=source_imgs,
        height=source_imgs[0].height,
        width=source_imgs[0].width,
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=3.5,
    ).images[0]
    image.save(output_path)
    print(f"编辑结果已保存至 {output_path}")


def perform_generation(input_img_paths, input_instruction, output_path, height=1024, width=1024):
    if "generation" not in AVAILABLE_ADAPTERS:
        raise gr.Error("未加载生成适配器，请检查控制台警告。")
    pipe.set_adapters(["generation"], adapter_weights=[1])  # 切换到生成适配器
    prefix = " It is generation task."
    source_imgs = []
    for path in input_img_paths:
        img = load_image(path)
        source_imgs.append(resizeinput(img))
    prompt = infer_vlm(input_img_paths, input_instruction, prefix)
    prompt = extract_gen_content(prompt)
    print(f"生成模式生成提示：{prompt}")

    image = pipe(
        images=source_imgs,
        height=height,
        width=width,
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=3.5,
    ).images[0]

    image.save(output_path)
    print(f"生成结果已保存至 {output_path}")


def process_edit_request(image_file_1, image_file_2, instruction):
    if not image_file_1 or not image_file_2:
        raise gr.Error("请上传两张图像。")
    if not instruction:
        raise gr.Error("请输入指令。")
    output_path = f"/tmp/{uuid.uuid4()}.png"
    input_img_paths = [image_file_1, image_file_2]
    perform_edit(input_img_paths, instruction, output_path)
    return output_path


def process_generation_request(image_file_1, image_file_2, instruction):
    if not image_file_1 or not image_file_2:
        raise gr.Error("请上传两张图像。")
    if not instruction:
        raise gr.Error("请输入指令。")
    output_path = f"/tmp/{uuid.uuid4()}.png"
    input_img_paths = [image_file_1, image_file_2]
    perform_generation(input_img_paths, instruction, output_path)
    return output_path


css = """
.text-center { text-align: center; }
.result-img img {
    max-height: 60vh !important;
    min-height: 30vh !important;
    width: auto !important;
    object-fit: contain;
}
.input-img img {
    max-height: 30vh !important;
    width: auto !important;
    object-fit: contain;
}
"""


with gr.Blocks(theme=gr.themes.Soft(), title="DreamOmni2", css=css) as demo:
    gr.HTML(
        """
        <h1 style="text-align:center; font-size:48px; font-weight:bold; margin-bottom:20px;">
            DreamOmni2：全能图像生成与编辑
        </h1>
        """
    )
    gr.Markdown(
        "上传参考图像，输入指令，然后运行所需流程。",
        elem_classes="text-center",
    )

    with gr.Tabs():
        with gr.Tab("编辑"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("⬆️ 上传图像，可点击或拖拽。")

                    with gr.Row():
                        edit_image_uploader_1 = gr.Image(
                            label="图像 1",
                            type="filepath",
                            interactive=True,
                            elem_classes="input-img",
                        )
                        edit_image_uploader_2 = gr.Image(
                            label="图像 2",
                            type="filepath",
                            interactive=True,
                            elem_classes="input-img",
                        )

                    edit_instruction_text = gr.Textbox(
                        label="指令",
                        lines=2,
                        placeholder="请描述如何利用参考图像编辑第一张图像...",
                    )
                    edit_run_button = gr.Button("开始编辑", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown(
                        "✏️ **编辑模式**：根据指令与参考图像修改现有图像。"
                        " 提示：若结果不符合预期，可再次点击 **开始编辑**。",
                    )
                    edit_output_image = gr.Image(
                        label="结果",
                        type="filepath",
                        elem_classes="result-img",
                    )

            gr.Markdown("## 编辑示例")
            gr.Examples(
                label="编辑示例",
                examples=[
                    [
                        "example_input/edit_tests/4/ref_0.jpg",
                        "example_input/edit_tests/4/ref_1.jpg",
                        "让第一张图像拥有与第二张图像相同的风格。",
                        "example_input/edit_tests/4/res.jpg",
                    ],
                    [
                        "example_input/edit_tests/5/ref_0.jpg",
                        "example_input/edit_tests/5/ref_1.jpg",
                        "让第一张图中的人物拥有第二张图人物的发型。",
                        "example_input/edit_tests/5/res.jpg",
                    ],
                    [
                        "example_input/edit_tests/src.jpg",
                        "example_input/edit_tests/ref.jpg",
                        "让第二张图中的女性站在第一张图的道路上。",
                        "example_input/edit_tests/edi_res.png",
                    ],
                    [
                        "example_input/edit_tests/1/ref_0.jpg",
                        "example_input/edit_tests/1/ref_1.jpg",
                        "将第一张图中的灯笼替换为第二张图中的狗。",
                        "example_input/edit_tests/1/res.jpg",
                    ],
                    [
                        "example_input/edit_tests/2/ref_0.jpg",
                        "example_input/edit_tests/2/ref_1.jpg",
                        "让第一张图中的西装变为第二张图的衣服。",
                        "example_input/edit_tests/2/res.jpg",
                    ],
                    [
                        "example_input/edit_tests/3/ref_0.jpg",
                        "example_input/edit_tests/3/ref_1.jpg",
                        "让第一张图拥有与第二张图相同的光照。",
                        "example_input/edit_tests/3/res.jpg",
                    ],
                    [
                        "example_input/edit_tests/6/ref_0.jpg",
                        "example_input/edit_tests/6/ref_1.jpg",
                        "让第一张图的文字采用第二张图的字体。",
                        "example_input/edit_tests/6/res.jpg",
                    ],
                    [
                        "example_input/edit_tests/7/ref_0.jpg",
                        "example_input/edit_tests/7/ref_1.jpg",
                        "让第一张图的汽车拥有第二张图老鼠的花纹。",
                        "example_input/edit_tests/7/res.jpg",
                    ],
                    [
                        "example_input/edit_tests/8/ref_0.jpg",
                        "example_input/edit_tests/8/ref_1.jpg",
                        "让第一张图的裙子具有第二张图的图案。",
                        "example_input/edit_tests/8/res.jpg",
                    ],
                ],
                inputs=[
                    edit_image_uploader_1,
                    edit_image_uploader_2,
                    edit_instruction_text,
                    edit_output_image,
                ],
                cache_examples=False,
            )

            edit_run_button.click(
                fn=process_edit_request,
                inputs=[edit_image_uploader_1, edit_image_uploader_2, edit_instruction_text],
                outputs=edit_output_image,
            )

        with gr.Tab("生成"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("⬆️ 上传图像，可点击或拖拽。")

                    with gr.Row():
                        gen_image_uploader_1 = gr.Image(
                            label="图像 1",
                            type="filepath",
                            interactive=True,
                            elem_classes="input-img",
                        )
                        gen_image_uploader_2 = gr.Image(
                            label="图像 2",
                            type="filepath",
                            interactive=True,
                            elem_classes="input-img",
                        )

                    gen_instruction_text = gr.Textbox(
                        label="指令",
                        lines=2,
                        placeholder="请描述希望基于参考图像生成的内容...",
                    )
                    gen_run_button = gr.Button("开始生成", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown(
                        "🖼️ **生成模式**：根据参考图像创建全新场景。"
                        " 提示：若结果不符合预期，可再次点击 **开始生成**。",
                    )
                    gen_output_image = gr.Image(
                        label="结果",
                        type="filepath",
                        elem_classes="result-img",
                    )

            gr.Markdown("## 生成示例")
            gr.Examples(
                label="生成示例",
                examples=[
                    [
                        "example_input/gen_tests/img1.jpg",
                        "example_input/gen_tests/img2.jpg",
                        "场景中，第一张图的角色站在左侧，第二张图的角色站在右侧，他们在飞船内部背景前握手。",
                        "example_input/gen_tests/gen_res.png",
                    ]
                ],
                inputs=[
                    gen_image_uploader_1,
                    gen_image_uploader_2,
                    gen_instruction_text,
                    gen_output_image,
                ],
                cache_examples=False,
            )

            gen_run_button.click(
                fn=process_generation_request,
                inputs=[gen_image_uploader_1, gen_image_uploader_2, gen_instruction_text],
                outputs=gen_output_image,
            )


if __name__ == "__main__":
    print("正在启动 Gradio Demo...")
    demo.launch(server_name=server_name, server_port=server_port)
