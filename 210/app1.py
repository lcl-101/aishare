import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import shutil
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


# ============================================================================
# Common Functions
# ============================================================================

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def extract_glb(state: dict, mesh_simplify: float, texture_size: int, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.

    Returns:
        str: The path to the extracted Gaussian file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


# ============================================================================
# Image to 3D Functions
# ============================================================================

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image = image_pipeline.preprocess_image(image)
    return processed_image


def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """
    Preprocess a list of input images.
    
    Args:
        images (List[Tuple[Image.Image, str]]): The input images.

    Returns:
        List[Image.Image]: The preprocessed images.
    """
    images = [image[0] for image in images]
    processed_images = [preprocess_image(image) for image in images]
    return processed_images


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split an image into multiple views.
    """
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]


def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["stochastic", "multidiffusion"],
    req: gr.Request,
) -> Tuple[dict, str]:
    """
    Convert an image to a 3D model.

    Args:
        image (Image.Image): The input image.
        multiimages (List[Tuple[Image.Image, str]]): The input images for multi-image generation.
        is_multiimage (bool): Whether the input is multi-image.
        seed (int): The random seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.
        multiimage_algo (Literal["stochastic", "multidiffusion"]): The algorithm for multi-image generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if is_multiimage and multiimages:
        images = [image[0] for image in multiimages]
        outputs = image_pipeline.run_multi_image(
            images,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )
    else:
        outputs = image_pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


# ============================================================================
# Text to 3D Functions
# ============================================================================

def text_to_3d(
    prompt: str,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    req: gr.Request,
) -> Tuple[dict, str]:
    """
    Convert a text prompt to a 3D model.

    Args:
        prompt (str): The text prompt.
        seed (int): The random seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    outputs = text_pipeline.run(
        prompt,
        seed=seed,
        formats=["gaussian", "mesh"],
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )

    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    # 使用 [TRELLIS](https://trellis3d.github.io/) 生成 3D 资产
    """)
    
    with gr.Tabs():
        # ====================================================================
        # Tab 1: Image to 3D
        # ====================================================================
        with gr.Tab("图像转 3D"):
            gr.Markdown("""
            * 上传图像并点击"生成"以创建 3D 资产。如果图像有透明通道，将被用作蒙版。否则，我们使用 `rembg` 去除背景。
            * 如果您对生成的 3D 资产满意，请点击"提取 GLB"以提取并下载 GLB 文件。
            """)
            
            with gr.Row():
                with gr.Column():
                    with gr.Tabs() as input_tabs:
                        with gr.Tab(label="单张图像", id=0) as single_image_input_tab:
                            image_prompt = gr.Image(label="图像输入", format="png", image_mode="RGBA", type="pil", height=300)
                        with gr.Tab(label="多张图像", id=1) as multiimage_input_tab:
                            multiimage_prompt = gr.Gallery(label="图像输入", format="png", type="pil", height=300, columns=3)
                            gr.Markdown("""
                                在不同的图像中输入对象的不同视角。
                                
                                *注意：这是一个未经专门模型训练的实验性算法。对于具有不同姿势或不一致细节的图像，可能无法产生最佳效果。*
                            """)
                    
                    with gr.Accordion(label="生成设置", open=False):
                        img_seed = gr.Slider(0, MAX_SEED, label="随机种子", value=0, step=1)
                        img_randomize_seed = gr.Checkbox(label="随机化种子", value=True)
                        gr.Markdown("阶段 1：稀疏结构生成")
                        with gr.Row():
                            img_ss_guidance_strength = gr.Slider(0.0, 10.0, label="引导强度", value=7.5, step=0.1)
                            img_ss_sampling_steps = gr.Slider(1, 50, label="采样步数", value=12, step=1)
                        gr.Markdown("阶段 2：结构化潜在空间生成")
                        with gr.Row():
                            img_slat_guidance_strength = gr.Slider(0.0, 10.0, label="引导强度", value=3.0, step=0.1)
                            img_slat_sampling_steps = gr.Slider(1, 50, label="采样步数", value=12, step=1)
                        multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="多图像算法", value="stochastic")

                    img_generate_btn = gr.Button("生成")
                    
                    with gr.Accordion(label="GLB 提取设置", open=False):
                        img_mesh_simplify = gr.Slider(0.9, 0.98, label="简化程度", value=0.95, step=0.01)
                        img_texture_size = gr.Slider(512, 2048, label="纹理大小", value=1024, step=512)
                    
                    with gr.Row():
                        img_extract_glb_btn = gr.Button("提取 GLB", interactive=False)
                        img_extract_gs_btn = gr.Button("提取高斯模型", interactive=False)
                    gr.Markdown("""
                                *注意：高斯文件可能非常大（约 50MB），显示和下载需要一些时间。*
                                """)

                with gr.Column():
                    img_video_output = gr.Video(label="生成的 3D 资产", autoplay=True, loop=True, height=300)
                    img_model_output = LitModel3D(label="提取的 GLB/高斯模型", exposure=10.0, height=300)
                    
                    with gr.Row():
                        img_download_glb = gr.DownloadButton(label="下载 GLB", interactive=False)
                        img_download_gs = gr.DownloadButton(label="下载高斯模型", interactive=False)  
            
            is_multiimage = gr.State(False)
            img_output_buf = gr.State()

            # Example images
            with gr.Row() as single_image_example:
                examples = gr.Examples(
                    examples=[
                        f'assets/example_image/{image}'
                        for image in os.listdir("assets/example_image")
                    ],
                    inputs=[image_prompt],
                    fn=preprocess_image,
                    outputs=[image_prompt],
                    run_on_click=True,
                    examples_per_page=64,
                )
            with gr.Row(visible=False) as multiimage_example:
                examples_multi = gr.Examples(
                    examples=prepare_multi_example(),
                    inputs=[image_prompt],
                    fn=split_image,
                    outputs=[multiimage_prompt],
                    run_on_click=True,
                    examples_per_page=8,
                )

            # Image to 3D Handlers
            single_image_input_tab.select(
                lambda: tuple([False, gr.Row.update(visible=True), gr.Row.update(visible=False)]),
                outputs=[is_multiimage, single_image_example, multiimage_example]
            )
            multiimage_input_tab.select(
                lambda: tuple([True, gr.Row.update(visible=False), gr.Row.update(visible=True)]),
                outputs=[is_multiimage, single_image_example, multiimage_example]
            )
            
            image_prompt.upload(
                preprocess_image,
                inputs=[image_prompt],
                outputs=[image_prompt],
            )
            multiimage_prompt.upload(
                preprocess_images,
                inputs=[multiimage_prompt],
                outputs=[multiimage_prompt],
            )

            img_generate_btn.click(
                get_seed,
                inputs=[img_randomize_seed, img_seed],
                outputs=[img_seed],
            ).then(
                image_to_3d,
                inputs=[image_prompt, multiimage_prompt, is_multiimage, img_seed, img_ss_guidance_strength, img_ss_sampling_steps, img_slat_guidance_strength, img_slat_sampling_steps, multiimage_algo],
                outputs=[img_output_buf, img_video_output],
            ).then(
                lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
                outputs=[img_extract_glb_btn, img_extract_gs_btn],
            )

            img_video_output.clear(
                lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
                outputs=[img_extract_glb_btn, img_extract_gs_btn],
            )

            img_extract_glb_btn.click(
                extract_glb,
                inputs=[img_output_buf, img_mesh_simplify, img_texture_size],
                outputs=[img_model_output, img_download_glb],
            ).then(
                lambda: gr.Button(interactive=True),
                outputs=[img_download_glb],
            )
            
            img_extract_gs_btn.click(
                extract_gaussian,
                inputs=[img_output_buf],
                outputs=[img_model_output, img_download_gs],
            ).then(
                lambda: gr.Button(interactive=True),
                outputs=[img_download_gs],
            )

            img_model_output.clear(
                lambda: gr.Button(interactive=False),
                outputs=[img_download_glb],
            )
        
        # ====================================================================
        # Tab 2: Text to 3D
        # ====================================================================
        with gr.Tab("文本转 3D"):
            gr.Markdown("""
            * 输入文本提示并点击"生成"以创建 3D 资产。
            * 如果您对生成的 3D 资产满意，请点击"提取 GLB"以提取并下载 GLB 文件。
            """)
            
            with gr.Row():
                with gr.Column():
                    text_prompt = gr.Textbox(label="文本提示", lines=5)
                    
                    with gr.Accordion(label="生成设置", open=False):
                        text_seed = gr.Slider(0, MAX_SEED, label="随机种子", value=0, step=1)
                        text_randomize_seed = gr.Checkbox(label="随机化种子", value=True)
                        gr.Markdown("阶段 1：稀疏结构生成")
                        with gr.Row():
                            text_ss_guidance_strength = gr.Slider(0.0, 10.0, label="引导强度", value=7.5, step=0.1)
                            text_ss_sampling_steps = gr.Slider(1, 50, label="采样步数", value=25, step=1)
                        gr.Markdown("阶段 2：结构化潜在空间生成")
                        with gr.Row():
                            text_slat_guidance_strength = gr.Slider(0.0, 10.0, label="引导强度", value=7.5, step=0.1)
                            text_slat_sampling_steps = gr.Slider(1, 50, label="采样步数", value=25, step=1)

                    text_generate_btn = gr.Button("生成")
                    
                    with gr.Accordion(label="GLB 提取设置", open=False):
                        text_mesh_simplify = gr.Slider(0.9, 0.98, label="简化程度", value=0.95, step=0.01)
                        text_texture_size = gr.Slider(512, 2048, label="纹理大小", value=1024, step=512)
                    
                    with gr.Row():
                        text_extract_glb_btn = gr.Button("提取 GLB", interactive=False)
                        text_extract_gs_btn = gr.Button("提取高斯模型", interactive=False)
                    gr.Markdown("""
                                *注意：高斯文件可能非常大（约 50MB），显示和下载需要一些时间。*
                                """)

                with gr.Column():
                    text_video_output = gr.Video(label="生成的 3D 资产", autoplay=True, loop=True, height=300)
                    text_model_output = LitModel3D(label="提取的 GLB/高斯模型", exposure=10.0, height=300)
                    
                    with gr.Row():
                        text_download_glb = gr.DownloadButton(label="下载 GLB", interactive=False)
                        text_download_gs = gr.DownloadButton(label="下载高斯模型", interactive=False)  
            
            text_output_buf = gr.State()

            # Text to 3D Handlers
            text_generate_btn.click(
                get_seed,
                inputs=[text_randomize_seed, text_seed],
                outputs=[text_seed],
            ).then(
                text_to_3d,
                inputs=[text_prompt, text_seed, text_ss_guidance_strength, text_ss_sampling_steps, text_slat_guidance_strength, text_slat_sampling_steps],
                outputs=[text_output_buf, text_video_output],
            ).then(
                lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
                outputs=[text_extract_glb_btn, text_extract_gs_btn],
            )

            text_video_output.clear(
                lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
                outputs=[text_extract_glb_btn, text_extract_gs_btn],
            )

            text_extract_glb_btn.click(
                extract_glb,
                inputs=[text_output_buf, text_mesh_simplify, text_texture_size],
                outputs=[text_model_output, text_download_glb],
            ).then(
                lambda: gr.Button(interactive=True),
                outputs=[text_download_glb],
            )
            
            text_extract_gs_btn.click(
                extract_gaussian,
                inputs=[text_output_buf],
                outputs=[text_model_output, text_download_gs],
            ).then(
                lambda: gr.Button(interactive=True),
                outputs=[text_download_gs],
            )

            text_model_output.clear(
                lambda: gr.Button(interactive=False),
                outputs=[text_download_glb],
            )

    # Global handlers
    demo.load(start_session)
    demo.unload(end_session)


# Launch the Gradio app
if __name__ == "__main__":
    # Load both pipelines
    image_pipeline = TrellisImageTo3DPipeline.from_pretrained("checkpoints/TRELLIS-image-large")
    image_pipeline.cuda()
    
    text_pipeline = TrellisTextTo3DPipeline.from_pretrained("checkpoints/TRELLIS-text-xlarge")
    text_pipeline.cuda()
    
    demo.launch(server_name="0.0.0.0")
