import os
import shlex
# import spaces
import subprocess
# def install_cuda_toolkit():
#     CUDA_TOOLKIT_URL = "https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
#     CUDA_TOOLKIT_FILE = "/tmp/%s" % os.path.basename(CUDA_TOOLKIT_URL)
#     subprocess.call(["wget", "-q", CUDA_TOOLKIT_URL, "-O", CUDA_TOOLKIT_FILE])
#     subprocess.call(["chmod", "+x", CUDA_TOOLKIT_FILE])
#     subprocess.call([CUDA_TOOLKIT_FILE, "--silent", "--toolkit"])

#     os.environ["CUDA_HOME"] = "/usr/local/cuda"
#     os.environ["PATH"] = "%s/bin:%s" % (os.environ["CUDA_HOME"], os.environ["PATH"])
#     os.environ["LD_LIBRARY_PATH"] = "%s/lib:%s" % (
#         os.environ["CUDA_HOME"],
#         "" if "LD_LIBRARY_PATH" not in os.environ else os.environ["LD_LIBRARY_PATH"],
#     )
#     os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
# # install_cuda_toolkit()
# # os.system("pip list | grep torch")
# # os.system('nvcc -V')
# print("cd /home/user/app/step1x3d_texture/differentiable_renderer/ && python setup.py install")
# os.system("cd /home/user/app/step1x3d_texture/differentiable_renderer/ && python setup.py install")

# subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"), check=True)
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
)
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face


parser = argparse.ArgumentParser()
parser.add_argument(
    "--geometry_model", type=str, default="Step1X-3D-Geometry-Label-1300m"
)
parser.add_argument(
    "--texture_model", type=str, default="Step1X-3D-Texture"
)
parser.add_argument("--cache_dir", type=str, default="cache")
args = parser.parse_args()

os.makedirs(args.cache_dir, exist_ok=True)

geometry_model = Step1X3DGeometryPipeline.from_pretrained(
    "checkpoints/Step1X-3D", subfolder=args.geometry_model
).to("cuda")

texture_model = Step1X3DTexturePipeline.from_pretrained("checkpoints/Step1X-3D", subfolder=args.texture_model)


# @spaces.GPU(duration=240)
def generate_func(
    input_image_path, guidance_scale, inference_steps, max_facenum, symmetry, edge_type
):
    # geometry_model = geometry_model.to("cuda")
    if "Label" in args.geometry_model:
        symmetry_values = ["x", "asymmetry"]
        out = geometry_model(
            input_image_path,
            label={"symmetry": symmetry_values[int(symmetry)], "edge_type": edge_type},
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
                choices=["symmetry", "asymmetry"],
                label="Symmetry Type",
                value="symmetry",
                type="index",
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

# demo.launch(ssr_mode=False)
demo.launch(server_name="0.0.0.0")
