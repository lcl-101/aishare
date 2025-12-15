import os
import time
import uuid
import torch
import trimesh
import argparse
import numpy as np
import gradio as gr
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
    Step1X3DTextureConfig,
)
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face


def generate_func(
    input_image_path, guidance_scale, inference_steps, max_facenum, symmetry, edge_type
):
    if "Label" in args.geometry_model:
        out = geometry_model(
            input_image_path,
            label={"symmetry": symmetry, "edge_type": edge_type},
            guidance_scale=float(guidance_scale),
            octree_resolution=384,
            max_facenum=int(max_facenum),
            num_inference_steps=int(inference_steps),
        )
    else:
        out = geometry_model(
            input_image_path,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(inference_steps),
            max_facenum=int(max_facenum),
        )

    save_name = str(uuid.uuid4())
    print(save_name)
    geometry_save_path = f"{args.cache_dir}/{save_name}.glb"
    geometry_mesh = out.mesh[0]
    geometry_mesh.export(geometry_save_path)

    geometry_mesh = remove_degenerate_face(geometry_mesh)
    geometry_mesh = reduce_face(geometry_mesh)
    textured_mesh = texture_model(input_image_path, geometry_mesh)
    textured_save_path = f"{args.cache_dir}/{save_name}-textured.glb"
    textured_mesh.export(textured_save_path)

    torch.cuda.empty_cache()
    print("Generate finish")
    return geometry_save_path, textured_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry_model", type=str, default="Step1X-3D-Geometry-Label-1300m"
    )
    parser.add_argument(
        "--texture_model", type=str, default="Step1X-3D-Texture"
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="checkpoints/Step1X-3D",
        help="Local checkpoint root folder (contains model subfolders like Step1X-3D-Geometry-Label-1300m, Step1X-3D-Texture, etc.)",
    )
    parser.add_argument(
        "--sdxl_base",
        type=str,
        default="checkpoints/stable-diffusion-xl-base-1.0",
        help="Local SDXL base model path",
    )
    parser.add_argument(
        "--sdxl_vae",
        type=str,
        default="checkpoints/sdxl-vae-fp16-fix",
        help="Local SDXL VAE model path",
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    checkpoint_root = os.path.abspath(args.checkpoint_root)
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(
            f"Checkpoint root not found: {checkpoint_root}. "
            "Expected something like checkpoints/Step1X-3D with model subfolders inside."
        )

    geometry_model = Step1X3DGeometryPipeline.from_pretrained(
        checkpoint_root, subfolder=args.geometry_model
    ).to("cuda")

    # Manually construct texture pipeline config with local model paths
    texture_config = Step1X3DTextureConfig()
    texture_config.base_model = os.path.abspath(args.sdxl_base)
    texture_config.vae_model = os.path.abspath(args.sdxl_vae)
    texture_config.adapter_path = os.path.join(checkpoint_root, args.texture_model)
    texture_model = Step1X3DTexturePipeline(texture_config)

    with gr.Blocks(title="Step1X-3D demo") as demo:
        gr.Markdown("# Step1X-3D")
        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(label="Image", type="filepath")
                guidance_scale = gr.Number(label="Guidance Scale", value="7.5")
                inference_steps = gr.Slider(
                    label="Inferece Steps", minimum=1, maximum=100, value=50
                )
                max_facenum = gr.Number(label="Max Face Num", value="400000")
                symmetry = gr.Radio(
                    choices=["x", "asymmetry"],
                    label="Symmetry Type",
                    value="x",
                    type="value",
                )
                edge_type = gr.Radio(
                    choices=["sharp", "normal", "smooth"],
                    label="Edge Type",
                    value="sharp",
                    type="value",
                )
                btn = gr.Button("Start")
            with gr.Column(scale=4):
                textured_preview = gr.Model3D(label="Textured", height=380)
                geometry_preview = gr.Model3D(label="Geometry", height=380)
            with gr.Column(scale=1):
                gr.Examples(
                    examples=[
                        ["examples/images/000.png"],
                        ["examples/images/001.png"],
                        ["examples/images/004.png"],
                        ["examples/images/008.png"],
                        ["examples/images/028.png"],
                        ["examples/images/032.png"],
                        ["examples/images/061.png"],
                        ["examples/images/107.png"],
                    ],
                    inputs=[input_image],
                    cache_examples=False,
                )

        btn.click(
            generate_func,
            inputs=[
                input_image,
                guidance_scale,
                inference_steps,
                max_facenum,
                symmetry,
                edge_type,
            ],
            outputs=[geometry_preview, textured_preview],
        )

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port)
