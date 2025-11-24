import argparse
import os
import tempfile
from typing import Any, Optional, Tuple

import cv2
import gradio as gr  
import numpy as np
import pyrootutils
import torch
import trimesh

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from tools.vis_utils import visualize_sample_together


DEFAULT_CHECKPOINT = os.path.join(
    "checkpoints", "sam-3d-body-dinov3", "model.ckpt"
)
DEFAULT_MHR_ASSET = os.path.join(
    "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt"
)

ALL_PERSONS_LABEL = "全部人物"


def _combine_meshes(outputs, template_faces):
    """Combine all predicted meshes into a single vertex/face array."""
    if not outputs:
        return None, None

    vertices_list = []
    faces_list = []
    vertex_offset = 0
    for person_output in outputs:
        verts = person_output["pred_vertices"] + person_output["pred_cam_t"]
        vertices_list.append(verts)
        faces_list.append(template_faces + vertex_offset)
        vertex_offset += verts.shape[0]

    vertices = np.concatenate(vertices_list, axis=0)
    faces = np.concatenate(faces_list, axis=0)
    # Rotate meshes so the person faces the viewer (align camera -Z with viewer -Z)
    rot = trimesh.transformations.rotation_matrix(
        np.deg2rad(-90), [1.0, 0.0, 0.0]
    )
    vertices = trimesh.transformations.transform_points(vertices, rot)
    return vertices, faces


def _export_mesh_files(vertices: np.ndarray, faces: np.ndarray) -> Tuple[str, str]:
    """Export combined mesh to OBJ (download) and GLB (viewer)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    tmp_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".obj")
    tmp_obj.close()
    mesh.export(tmp_obj.name, file_type="obj")

    tmp_glb = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    tmp_glb.close()
    mesh.export(tmp_glb.name, file_type="glb")

    return tmp_obj.name, tmp_glb.name


def _resolve_path(value: str, env_key: str, fallback: str = "") -> str:
    """Return CLI value or fall back to environment variable or fallback."""
    candidate = value or os.environ.get(env_key, "") or fallback
    return os.path.expanduser(candidate)


def _maybe_build_detector(name: str, device: torch.device, path: str):
    if not name:
        return None
    from tools.build_detector import HumanDetector

    print(f"Initializing detector '{name}' from '{path or 'default weights'}'")
    return HumanDetector(name=name, device=device, path=path)


def _maybe_build_segmentor(name: str, device: torch.device, path: str):
    if not name or not path:
        if name and not path:
            print("Segmentor name provided but path missing; skipping SAM segmentor initialization.")
        return None
    from tools.build_sam import HumanSegmentor

    print(f"Initializing segmentor '{name}' from '{path}'")
    return HumanSegmentor(name=name, device=device, path=path)


def _maybe_build_fov_estimator(name: str, device: torch.device, path: str):
    if not name:
        return None
    from tools.build_fov_estimator import FOVEstimator

    print(f"Initializing FOV estimator '{name}' from '{path or 'default weights'}'")
    return FOVEstimator(name=name, device=device, path=path)


def build_estimator(args) -> SAM3DBodyEstimator:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_path(
        args.checkpoint_path,
        "SAM3D_CHECKPOINT_PATH",
        DEFAULT_CHECKPOINT,
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    mhr_path = _resolve_path(
        args.mhr_path,
        "SAM3D_MHR_PATH",
        DEFAULT_MHR_ASSET,
    )
    detector_path = _resolve_path(args.detector_path, "SAM3D_DETECTOR_PATH")
    segmentor_path = _resolve_path(args.segmentor_path, "SAM3D_SEGMENTOR_PATH")
    fov_path = _resolve_path(args.fov_path, "SAM3D_FOV_PATH")

    model, model_cfg = load_sam_3d_body(
        checkpoint_path=checkpoint_path,
        device=device,
        mhr_path=mhr_path,
    )

    human_detector = _maybe_build_detector(args.detector_name, device, detector_path)
    human_segmentor = _maybe_build_segmentor(args.segmentor_name, device, segmentor_path)
    fov_estimator = _maybe_build_fov_estimator(args.fov_name, device, fov_path)

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )


class GradioSAM3DBodyApp:
    def __init__(self, args):
        self.args = args
        self.estimator = build_estimator(args)
        self.current_outputs = []
        self.current_image_rgb = None
        self.person_choice_map = {ALL_PERSONS_LABEL: None}
        self.last_use_mask = args.use_mask
        self.last_inference_type = args.inference_type

    def process_image(
        self,
        image,
        bbox_thresh: float,
        use_mask: bool,
        inference_type: str,
        person_choice: str,
    ) -> Tuple[Optional[np.ndarray], str, Optional[str], Any]:
        if image is None:
            dropdown_update = gr.update(
                choices=[ALL_PERSONS_LABEL], value=ALL_PERSONS_LABEL
            )
            return None, "请先上传一张图片。", None, dropdown_update

        image_rgb = np.array(image.convert("RGB"))
        try:
            outputs = self.estimator.process_one_image(
                image_rgb,
                bbox_thr=bbox_thresh,
                use_mask=use_mask,
                inference_type=inference_type,
            )
        except Exception as exc:  # noqa: BLE001
            dropdown_update = gr.update(
                choices=[ALL_PERSONS_LABEL], value=ALL_PERSONS_LABEL
            )
            return None, f"推理失败：{exc}", None, dropdown_update

        if len(outputs) == 0:
            self.current_outputs = []
            self.current_image_rgb = None
            dropdown_update = gr.update(
                choices=[ALL_PERSONS_LABEL], value=ALL_PERSONS_LABEL
            )
            return (
                None,
                "未检测到人体，请尝试降低阈值或更换图片。",
                None,
                dropdown_update,
            )

        input_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        self.current_outputs = outputs
        self.current_image_rgb = image_rgb

        person_choices = self._build_person_choices(outputs)
        selected_choice = (
            person_choice if person_choice in self.person_choice_map else ALL_PERSONS_LABEL
        )
        dropdown_update = gr.update(
            choices=person_choices,
            value=selected_choice,
        )

        self.last_use_mask = use_mask
        self.last_inference_type = inference_type

        filtered_outputs = self._filter_outputs(outputs, selected_choice)
        rendered_rgb, status, glb_path = self._render_selection(
            input_bgr,
            filtered_outputs,
            len(outputs),
            selected_choice,
            use_mask,
            inference_type,
        )

        return rendered_rgb, status, glb_path, dropdown_update

    def update_selection(
        self, person_choice: str
    ) -> Tuple[Optional[np.ndarray], str, Optional[str]]:
        if self.current_image_rgb is None or len(self.current_outputs) == 0:
            return None, "请先运行一次推理，然后再选择人物。", None

        choice = (
            person_choice if person_choice in self.person_choice_map else ALL_PERSONS_LABEL
        )
        filtered_outputs = self._filter_outputs(self.current_outputs, choice)
        input_bgr = cv2.cvtColor(self.current_image_rgb, cv2.COLOR_RGB2BGR)
        rendered_rgb, status, glb_path = self._render_selection(
            input_bgr,
            filtered_outputs,
            len(self.current_outputs),
            choice,
            self.last_use_mask,
            self.last_inference_type,
        )
        return rendered_rgb, status, glb_path

    def _build_person_choices(self, outputs):
        choices = [ALL_PERSONS_LABEL]
        self.person_choice_map = {ALL_PERSONS_LABEL: None}
        for idx, person_output in enumerate(outputs):
            bbox = person_output["bbox"]
            label = f"人物 {idx} (x={bbox[0]:.0f}, y={bbox[1]:.0f})"
            self.person_choice_map[label] = idx
            choices.append(label)
        return choices

    def _filter_outputs(self, outputs, choice):
        if len(outputs) == 0:
            return []
        idx = self.person_choice_map.get(choice)
        if idx is None:
            return outputs
        if 0 <= idx < len(outputs):
            return [outputs[idx]]
        return []

    def _render_selection(
        self,
        input_bgr,
        outputs,
        total_count: int,
        choice_label: str,
        use_mask: bool,
        inference_type: str,
    ):
        if len(outputs) == 0:
            status = (
                "当前选择未匹配到人物。"
                f" 最近一次推理共检测到 {total_count} 人。"
            )
            return None, status, None

        rendered_bgr = visualize_sample_together(input_bgr, outputs, self.estimator.faces)
        rendered_rgb = cv2.cvtColor(rendered_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)

        status = f"推理完成：共检测到 {total_count} 人。"
        if choice_label != ALL_PERSONS_LABEL:
            status += f" 当前展示：{choice_label}。"
        if use_mask and self.estimator.sam is None:
            status += " SAM 分割模型未加载，已退回到仅使用检测框的模式。"
        if inference_type != "full":
            status += f" 当前推理模式：{inference_type}。"

        obj_path, glb_path = None, None
        vertices, faces = _combine_meshes(outputs, self.estimator.faces)
        if vertices is not None:
            obj_path, glb_path = _export_mesh_files(vertices, faces)

        return rendered_rgb, status, glb_path

    def build_ui(self):
        with gr.Blocks(title="SAM 3D Body 在线演示") as demo:
            gr.Markdown(
                """
                # SAM 3D Body 在线演示
                上传一张 RGB 图片，即可在线查看人体三维网格和关键点可视化效果。
                """
            )

            with gr.Row():
                image_input = gr.Image(label="输入图片", type="pil")
                image_output = gr.Image(label="渲染结果", interactive=False)

            with gr.Row():
                bbox_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=self.args.bbox_thresh,
                    label="检测阈值",
                )
                use_mask_checkbox = gr.Checkbox(
                    label="启用 SAM Mask（需要分割模型）",
                    value=self.args.use_mask,
                )
                inference_dropdown = gr.Dropdown(
                    choices=["full", "body", "hand"],
                    value=self.args.inference_type,
                    label="推理模式",
                )
                person_selector = gr.Dropdown(
                    choices=[ALL_PERSONS_LABEL],
                    value=ALL_PERSONS_LABEL,
                    label="选择人物",
                    interactive=True,
                )

            status_box = gr.Textbox(label="状态", interactive=False)
            mesh_view = gr.Model3D(label="3D 预览", clear_color=[0.05, 0.05, 0.05, 1])
            run_button = gr.Button("开始推理", variant="primary")

            run_button.click(
                fn=self.process_image,
                inputs=[
                    image_input,
                    bbox_slider,
                    use_mask_checkbox,
                    inference_dropdown,
                    person_selector,
                ],
                outputs=[
                    image_output,
                    status_box,
                    mesh_view,
                    person_selector,
                ],
            )

            person_selector.change(
                fn=self.update_selection,
                inputs=person_selector,
                outputs=[image_output, status_box, mesh_view],
            )

        return demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Gradio App",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        default="",
        type=str,
        help="Path to SAM 3D Body model checkpoint (defaults to bundled dinov3).",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR assets folder (or set SAM3D_MHR_PATH).",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Name of detector to use (empty to disable).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="",
        type=str,
        help="Name of SAM segmentor to use (empty to disable).",
    )
    parser.add_argument(
        "--fov_name",
        default="",
        type=str,
        help="Name of FOV estimator to use (empty to disable).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to detector weights directory (or set SAM3D_DETECTOR_PATH).",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to SAM2 project root (or set SAM3D_SEGMENTOR_PATH).",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path or HF hub id for FOV estimator (or set SAM3D_FOV_PATH).",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box threshold used for detection slider default.",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="Enable mask-conditioned prediction by default.",
    )
    parser.add_argument(
        "--inference_type",
        choices=["full", "body", "hand"],
        default="full",
        help="Inference mode passed to estimator.",
    )
    parser.add_argument(
        "--server_name",
        default="0.0.0.0",
        type=str,
        help="Host for Gradio app.",
    )
    parser.add_argument(
        "--server_port",
        default=7860,
        type=int,
        help="Port for Gradio app.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Launch Gradio with public sharing.",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Enable Gradio queue for concurrency control.",
    )
    parser.add_argument(
        "--max_threads",
        default=40,
        type=int,
        help="Maximum threads for torch operations (passed to torch.set_num_threads).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(args.max_threads)

    app = GradioSAM3DBodyApp(args)
    demo = app.build_ui()

    if args.queue:
        demo = demo.queue()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
