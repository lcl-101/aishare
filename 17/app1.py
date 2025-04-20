import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

import gradio as gr

# 下载检查点 
#snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

mask_predictor = AutoMasker(
    densepose_path="./ckpts/densepose",
    schp_path="./ckpts/schp",
)
densepose_predictor = DensePosePredictor(
    config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
    weights_path="./ckpts/densepose/model_final_162be9.pkl",
)
vt_model = LeffaModel(
    pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
    pretrained_model="./ckpts/virtual_tryon.pth",
)
vt_inference = LeffaInference(model=vt_model)

def leffa_predict(src_image_path, ref_image_path):
    src_image = Image.open(src_image_path)
    ref_image = Image.open(ref_image_path)
    src_image = resize_and_center(src_image, 768, 1024)
    ref_image = resize_and_center(ref_image, 768, 1024)

    src_image_array = np.array(src_image)
    ref_image_array = np.array(ref_image)

    # 生成虚拟试穿的掩码
    src_image = src_image.convert("RGB")
    mask = mask_predictor(src_image, "upper")["mask"]

    # 生成 DensePose 分割
    src_image_seg_array = densepose_predictor.predict_seg(src_image_array)
    densepose = Image.fromarray(src_image_seg_array)

    # 准备推理数据
    transform = LeffaTransform()

    data = {
        "src_image": [src_image],
        "ref_image": [ref_image],
        "mask": [mask],
        "densepose": [densepose],
    }
    data = transform(data)

    # 执行推理
    output = vt_inference(data)
    gen_image = output["generated_image"][0]
    return np.array(gen_image)

if __name__ == "__main__":
    title = "## Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation"
    link = "[📚 Paper](https://arxiv.org/abs/2412.08486) - [🤖 Code](https://github.com/franciszzj/Leffa) - [🔥 Demo](https://huggingface.co/spaces/franciszzj/Leffa) - [🤗 Model](https://huggingface.co/franciszzj/Leffa)"
    description = "Leffa 是一个统一的可控人像生成框架，能够精确操控外观（即虚拟试穿）和姿势（即姿势转移）。"

    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink, secondary_hue=gr.themes.colors.red)).queue() as demo:
        gr.Markdown(title)
        gr.Markdown(link)
        gr.Markdown(description)

        with gr.Tab("Control Appearance (Virtual Try-on)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Person Image")
                    vt_src_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Person Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=vt_src_image,
                        examples_per_page=5,
                        examples=[
                            "./ckpts/examples/person1/01350_00.jpg",
                            "./ckpts/examples/person1/01376_00.jpg",
                            "./ckpts/examples/person1/01416_00.jpg",
                            "./ckpts/examples/person1/05976_00.jpg",
                            "./ckpts/examples/person1/06094_00.jpg",
                        ],
                    )

                with gr.Column():
                    gr.Markdown("#### Garment Image")
                    vt_ref_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Garment Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=vt_ref_image,
                        examples_per_page=5,
                        examples=[
                            "./ckpts/examples/garment/01449_00.jpg",
                            "./ckpts/examples/garment/01486_00.jpg",
                            "./ckpts/examples/garment/01853_00.jpg",
                            "./ckpts/examples/garment/02070_00.jpg",
                            "./ckpts/examples/garment/03553_00.jpg",
                        ],
                    )

                with gr.Column():
                    gr.Markdown("#### Generated Image")
                    vt_gen_image = gr.Image(
                        label="Generated Image",
                        width=512,
                        height=512,
                    )

                    with gr.Row():
                        vt_gen_button = gr.Button("Generate")

                # 连接生成按钮到预测函数
                vt_gen_button.click(
                    fn=leffa_predict,
                    inputs=[vt_src_image, vt_ref_image],
                    outputs=[vt_gen_image],
                )

        demo.launch(share=True, server_name="0.0.0.0", server_port=7860, allowed_paths=["./ckpts/examples"])