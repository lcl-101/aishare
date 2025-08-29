# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import os
from pathlib import Path

import gradio as gr
import torch

from transformers import SiglipVisionModel, SiglipImageProcessor

# ---------------------------------------------------------------------------
# 自动本地模型路径设置
# 如果用户没有显式设置环境变量 (AE / FLUX_DEV / LORA / PROJECTION_MODEL)，
# 且本地 checkpoints 目录下已经存在对应文件，则自动填充，避免手动 export。
# 目录结构假定为：
# checkpoints/
#   FLUX.1-dev/
#       flux1-dev.safetensors
#       ae.safetensors
#   USO/
#       uso_flux_v1.0/
#           dit_lora.safetensors
#           projector.safetensors
# ---------------------------------------------------------------------------
def _auto_set_local_paths():
    base = Path("checkpoints")
    # Base FLUX model
    flux_dir = base / "FLUX.1-dev"
    uso_dir = base / "USO" / "uso_flux_v1.0"
    # Helper to resolve snapshot (HF cache copied) directory
    def _resolve_dir(p: Path):
        if not p.exists():
            return None
        snap_root = p / 'snapshots'
        if snap_root.exists():
            snaps = sorted(snap_root.glob('*'))
            if snaps:
                return snaps[0]
        return p

    def _first_exist(candidates: list[Path]):
        for c in candidates:
            r = _resolve_dir(c)
            if r and r.exists():
                return r
        return None

    clip_dir = _first_exist([
        base / 'CLIP',
        base / 'clip-vit-large-patch14',
        base / 'openai-clip-vit-large-patch14'
    ])
    t5_dir = _first_exist([
        base / 'T5',
        base / 'xflux_text_encoders',
        base / 'text_encoders'
    ])
    siglip_dir = _first_exist([
        base / 'SIGLIP',
        base / 'siglip-so400m-patch14-384',
        base / 'google-siglip-so400m-patch14-384'
    ])

    mapping = {
        "FLUX_DEV": flux_dir / "flux1-dev.safetensors",
        "AE": flux_dir / "ae.safetensors",
        "LORA": uso_dir / "dit_lora.safetensors",
        "PROJECTION_MODEL": uso_dir / "projector.safetensors",
        "CLIP": clip_dir,
        "T5": t5_dir,
        "SIGLIP": siglip_dir,
    }

    for env_key, path in mapping.items():
        if os.environ.get(env_key):
            continue
        if path and ((path.is_file()) or (path.is_dir())):
            os.environ[env_key] = str(path.resolve())
            print(f"[AUTO PATH] {env_key} -> {path}")
        else:
            print(f"[AUTO PATH WARNING] 期望的文件不存在: {path}")


# 在导入 / 创建 pipeline 前尝试自动设置 (需先于导入 USOPipeline)
_auto_set_local_paths()

# 现在再导入 pipeline，确保 util.configs 读取到已设置的环境变量
from uso.flux.pipeline import USOPipeline  # noqa: E402

print(f"[DEBUG] FLUX_DEV={os.environ.get('FLUX_DEV')}")


with open("assets/uso_text.svg", "r", encoding="utf-8") as svg_file:
    text_content = svg_file.read()

with open("assets/uso_logo.svg", "r", encoding="utf-8") as svg_file:
    logo_content = svg_file.read()

title = f"""
<div style="display: flex; align-items: center; justify-content: center;">
    <span style="transform: scale(0.7);margin-right: -5px;">{text_content}</span>    
    <span style="font-size: 1.8em;margin-left: -10px;font-weight: bold; font-family: Gill Sans;">by UXO Team</span>
    <span style="margin-left: 0px; transform: scale(0.85); display: inline-block;">{logo_content}</span>
</div>
""".strip()

badges_text = r"""
<div style="text-align: center; display: flex; justify-content: center; gap: 5px;">
<a href="https://github.com/bytedance/USO"><img src="https://img.shields.io/static/v1?label=GitHub&message=Code&color=green&logo=github"></a>
<a href="https://bytedance.github.io/USO/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-USO-yellow"></a>
<a href="https://arxiv.org/abs/2504.02160"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-USO-b31b1b.svg"></a>
<a href="https://huggingface.co/bytedance-research/USO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>
</div>
""".strip()

tips = """
**What is USO?**  🎨
USO is a unified style-subject optimized customization model and the latest addition to the UXO family (<a href='https://github.com/bytedance/USO' target='_blank'> USO</a> and <a href='https://github.com/bytedance/UNO' target='_blank'> UNO</a>). 
It can freely combine any subjects with any styles in any scenarios.

**How to use?**    💡
We provide step-by-step instructions in our <a href='https://github.com/bytedance/USO' target='_blank'> Github Repo</a>.
Additionally, try the examples provided below the demo to quickly get familiar with USO and spark your creativity!

<details>
<summary style="cursor: pointer; color: #d34c0e; font-weight: 500;">The model is trained on 1024x1024 resolution and supports 3 types of usage. 📌 Tips:</summary>

* **Only content img**: support following types:
  * Subject/Identity-driven (supports natural prompt, e.g., *A clock on the table.* *The woman near the sea.*, excels in producing **photorealistic portraits**)
  * Style edit (layout-preserved): *Transform the image into Ghibli style/Pixel style/Retro comic style/Watercolor painting style...*.
  * Style edit (layout-shift): *Ghibli style, the man on the beach.*.
* **Only style img**: Reference input style and generate anything following prompt. Excelling in this and further support multiple style references (in beta).
* **Content img + style img**: Place the content into the desired style. 
  * Layout-preserved: set prompt to **empty**.
  * Layout-shift: using natural prompt.</details>"""

star = r"""
If USO is helpful, please help to ⭐ our <a href='https://github.com/bytedance/USO' target='_blank'> Github Repo</a>. Thanks a lot!"""

def get_examples(examples_dir: str = "assets/examples") -> list:
    examples = Path(examples_dir)
    ans = []  
    for example in examples.iterdir():
        if not example.is_dir() or len(os.listdir(example)) == 0:
            continue
        with open(example / "config.json") as f:
            example_dict = json.load(f)


        example_list = []
        example_list.append(example_dict["prompt"])  # prompt

        for key in ["image_ref1", "image_ref2", "image_ref3"]:
            if key in example_dict:
                example_list.append(str(example / example_dict[key]))
            else:
                example_list.append(None)

        example_list.append(example_dict["seed"])
        ans.append(example_list)
    return ans


def create_demo(
    model_type: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
):
    pipeline = USOPipeline(
        model_type, device, offload, only_lora=True, lora_rank=128, hf_download=True
    )
    print("USOPipeline loaded successfully")
    # ---- Local SigLIP (optional) ----
    siglip_path = os.environ.get("SIGLIP")
    if siglip_path and os.path.exists(siglip_path):
        try:
            print(f"Loading SigLIP locally from {siglip_path}")
            siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path, local_files_only=True)
            siglip_model = SiglipVisionModel.from_pretrained(siglip_path, local_files_only=True)
            siglip_model.eval()
            siglip_model.to(device)
            pipeline.model.vision_encoder = siglip_model
            pipeline.model.vision_encoder_processor = siglip_processor
            print("SigLIP model loaded successfully (local)")
        except Exception as e:
            print(f"[SIGLIP WARNING] 本地 SigLIP 加载失败: {e}. 将跳过。")
    else:
        print("[SIGLIP NOTICE] 未找到本地 SigLIP (设置 SIGLIP 或放到 checkpoints/siglip-so400m-patch14-384)。已跳过。")

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(badges_text)
        gr.Markdown(tips)
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A beautiful woman.")
                with gr.Row():
                    image_prompt1 = gr.Image(
                        label="Content Reference Img", visible=True, interactive=True, type="pil"
                    )
                    image_prompt2 = gr.Image(
                        label="Style Reference Img", visible=True, interactive=True, type="pil"
                    )
                    image_prompt3 = gr.Image(
                        label="Extra Style Reference Img (Beta)", visible=True, interactive=True, type="pil"
                    )

                with gr.Row():
                    with gr.Row():
                        width = gr.Slider(
                            512, 1536, 1024, step=16, label="Generation Width"
                        )
                        height = gr.Slider(
                            512, 1536, 1024, step=16, label="Generation Height"
                        )
                with gr.Row():
                    with gr.Row():
                        keep_size = gr.Checkbox(
                            label="Keep input size",
                            value=False,
                            interactive=True
                        )
                    with gr.Column():
                        gr.Markdown("Set it to True if you only need style editing or want to keep the layout.")

                with gr.Accordion("Advanced Options", open=True):
                    with gr.Row():
                        num_steps = gr.Slider(
                            1, 50, 25, step=1, label="Number of steps"
                        )
                        guidance = gr.Slider(
                            1.0, 5.0, 4.0, step=0.1, label="Guidance", interactive=True
                        )
                        content_long_size = gr.Slider(
                            0, 1024, 512, step=16, label="Content reference size"
                        )                        
                        seed = gr.Number(-1, label="Seed (-1 for random)")

                generate_btn = gr.Button("Generate")
                gr.Markdown(star)

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                download_btn = gr.File(
                    label="Download full-resolution", type="filepath", interactive=False
                )

            inputs = [
                prompt,
                image_prompt1,
                image_prompt2,
                image_prompt3,
                seed,                     
                width,
                height,
                guidance,
                num_steps,
                keep_size,
                content_long_size,
            ]
            generate_btn.click(
                fn=pipeline.gradio_generate,
                inputs=inputs,
                outputs=[output_image, download_btn],
            )   

        # example_text = gr.Text("", visible=False, label="Case For:")
        examples = get_examples("./assets/gradio_examples")

        gr.Examples(
            examples=examples,
            inputs=[
                prompt,
                image_prompt1,
                image_prompt2,
                image_prompt3,
                seed,
            ],
            # cache_examples='lazy',
            outputs=[output_image, download_btn],
            fn=pipeline.gradio_generate,
            label='row 1-4: identity/subject-driven; row 5-7: style-subject-driven; row 8-9: style-driven; row 10-12: multi-style-driven task; row 13: txt2img',
            examples_per_page=15
        )

    return demo


if __name__ == "__main__":
    from typing import Literal

    from transformers import HfArgumentParser

    @dataclasses.dataclass
    class AppArgs:
        name: Literal["flux-dev", "flux-dev-fp8", "flux-schnell", "flux-krea-dev"] = "flux-dev"
        device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
        offload: bool = dataclasses.field(
            default=False,
            metadata={
                "help": "If True, sequantial offload the models(ae, dit, text encoder) to CPU if not used."
            },
        )
        port: int = 7860

    parser = HfArgumentParser([AppArgs])
    args_tuple = parser.parse_args_into_dataclasses()  # type: tuple[AppArgs]
    args = args_tuple[0]

    demo = create_demo(args.name, args.device, args.offload)
    # 监听所有网卡以便外部访问
    demo.launch(server_port=args.port, server_name="0.0.0.0")
