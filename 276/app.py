import os
import numpy as np
import torch
import gradio as gr  
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm


class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")

        # ASR model for prompt text recognition
        self.asr_model_id = str(Path.cwd() / "checkpoints" / "SenseVoiceSmall")
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level='DEBUG',
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

        # TTS model (lazy init)
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = os.getenv(
            "VOXCPM_MODEL_DIR",
            str(Path.cwd() / "checkpoints" / "VoxCPM1.5"),
        )
        self.zipenhancer_local_model_dir = str(Path.cwd() / "checkpoints" / "speech_zipenhancer_ans_multiloss_16k_base")

    # ---------- Model helpers ----------
    def _resolve_model_dir(self) -> str:
        model_dir = os.path.abspath(self.default_local_model_dir)
        if os.path.isdir(model_dir):
            return model_dir
        raise FileNotFoundError(
            f"本地模型目录不存在：{model_dir}，请先将 VoxCPM 模型放到该目录或设置环境变量 VOXCPM_MODEL_DIR。"
        )

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("Model not loaded, initializing...")
        model_dir = self._resolve_model_dir()
        print(f"Using model dir: {model_dir}")
        # 默认禁用降噪，避免自动下载 ZipEnhancer 模型；如需启用，请在本地提供路径并设置 enable_denoiser=True。
        # optimize=False 禁用 torch.compile，避免首次编译卡住
        self.voxcpm_model = voxcpm.VoxCPM(
            voxcpm_model_path=model_dir,
            zipenhancer_model_path=self.zipenhancer_local_model_dir,
            enable_denoiser=True,
            optimize=False,
        )
        print("Model loaded successfully.")
        return self.voxcpm_model

    # ---------- Functional endpoints ----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        text = res[0]["text"].split('|>')[-1]
        return text

    def generate_tts_audio(
        self,
        text_input: str,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate speech from text using VoxCPM; optional reference audio for voice style guidance.
        Returns (sample_rate, waveform_numpy)
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("请输入要合成的文本。")

        prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
        prompt_text = prompt_text_input if prompt_text_input else None

        print(f"Generating audio for text: '{text[:60]}...'")
        wav = current_model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=denoise,
        )
        return (current_model.tts_model.sample_rate, wav)


# ---------- UI Builders ----------

def create_demo_interface(demo: VoxCPMDemo):
    """构建 VoxCPM 演示的 Gradio 界面。"""
    # static assets (logo path)
    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css="""
        .logo-container {
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
        .logo-container img {
            height: 80px;
            width: auto;
            max-width: 200px;
            display: inline-block;
        }
        /* Bold accordion labels */
        #acc_quick details > summary,
        #acc_tips details > summary {
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }
        /* Bold labels for specific checkboxes */
        #chk_denoise label,
        #chk_denoise span,
        #chk_normalize label,
        #chk_normalize span {
            font-weight: 600;
        }
        """
    ) as interface:
        # Header logo
        gr.HTML('<div class="logo-container"><img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo"></div>')

        # Quick Start
        with gr.Accordion("📋 快速入门", open=False, elem_id="acc_quick"):
            gr.Markdown("""
            ### 使用步骤
            1. （可选）上传或录制一段提示语音，用于指定音色/情感。
            2. （可选）输入提示语音的文字稿；如留空，可自动识别后再人工校对。
            3. 输入需要合成的目标文本。
            4. 点击“生成语音”按钮，即可得到音频。
            """)

        # Pro Tips
        with gr.Accordion("💡 使用建议", open=False, elem_id="acc_tips"):
            gr.Markdown("""
            ### 提示语音降噪
            - 开启：使用 ZipEnhancer 去噪，采样率限制 16kHz，音色还原上限略低。
            - 关闭：保留原始背景信息，最高支持 44.1kHz，更适合高保真克隆。

            ### 文本正则化
            - 开启：使用 wetext 库做常规文本规范化。
            - 关闭：使用 VoxCPM 内置理解能力，支持音素输入（例：中文 {ni3}{hao3}；英文 {HH AH0 L OW1}）。

            ### CFG 值
            - 适当调低：提示语音过于夸张或长文本不稳定时。
            - 适当调高：需要更贴合提示音频或极短文本不稳定时。

            ### 推理时间步
            - 调低：加快速度。
            - 调高：提升音质。
            """)

        # Main controls
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    sources=["upload", 'microphone'],
                    type="filepath",
                    label="提示语音（可选，可上传或录制）",
                    value="./examples/example.wav",
                )
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="提示语音降噪",
                    elem_id="chk_denoise",
                    info="使用 ZipEnhancer 对提示音频做降噪，开启后采样率限制为 16kHz。"
                )
                with gr.Row():
                    prompt_text = gr.Textbox(
                        value="Just by listening a few minutes a day, you'll be able to eliminate negative thoughts by conditioning your mind to be more positive.",
                        label="提示文本",
                        placeholder="如果提供了提示语音，请输入对应文本；留空可自动识别后再修改。"
                    )
                run_btn = gr.Button("生成语音", variant="primary")

            with gr.Column():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG 值（引导强度）",
                    info="高值更贴合提示音色，低值更具创意"
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="推理时间步数",
                    info="越高音质越好但更慢，越低越快"
                )
                with gr.Row():
                    text = gr.Textbox(
                        value="VoxCPM is an end-to-end high-fidelity text-to-speech model.",
                        label="目标文本",
                    )
                gr.Examples(
                    examples=[
                        ["VoxCPM is an end-to-end high-fidelity text-to-speech model."],
                        ["它是基于目前最先进的 Video Diffusion Transformer 架构——也就是和 Sora，通义万相同源的技术。"],
                        ["它是基于目前最先进的 Video Diffusion Transformer 架构——也就是和 Sora，通义万{xiang4}同源的技术。"],
                    ],
                    inputs=[text],
                    label="示例文本（点击填充）",
                )
                with gr.Row():
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label="文本正则化",
                        elem_id="chk_normalize",
                        info="开启后使用 wetext 进行文本规范化"
                    )
                audio_output = gr.Audio(label="输出音频")

        # Wiring
        run_btn.click(
            fn=demo.generate_tts_audio,
            inputs=[text, prompt_wav, prompt_text, cfg_value, inference_timesteps, DoNormalizeText, DoDenoisePromptAudio],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )
        prompt_wav.change(fn=demo.prompt_wav_recognition, inputs=[prompt_wav], outputs=[prompt_text])

    return interface


def run_demo(server_name: str = "0.0.0.0", server_port: int = 7860, show_error: bool = True):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    # Recommended to enable queue on Spaces for better throughput
    interface.queue(max_size=10, default_concurrency_limit=1).launch(server_name=server_name, server_port=server_port, show_error=show_error)


if __name__ == "__main__":
    run_demo()