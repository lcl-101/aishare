
# ==== OmniAvatar WebUI - 配置文件和参数初始化 ====
import yaml
from types import SimpleNamespace

# 自定义 ArgsNamespace 类，支持 dict 风格访问和属性访问
class ArgsNamespace(SimpleNamespace):
    """兼容性命名空间，支持 'key' in args 和 args.key 两种访问方式"""
    def __contains__(self, key):
        return hasattr(self, key)
    def __iter__(self):
        return iter(self.__dict__)
    def keys(self):
        return self.__dict__.keys()
    def items(self):
        return self.__dict__.items()
    def get(self, key, default=None):
        return getattr(self, key, default)

# 加载推理配置文件
with open("configs/inference.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

# 将 model_config 参数合并到顶层，确保模型结构参数正确
if "model_config" in config_dict and config_dict["model_config"]:
    config_dict.update(config_dict["model_config"])

config_dict['config'] = "configs/inference.yaml"
args = ArgsNamespace(**config_dict)

# 全局注入 args 对象，供模型代码使用
import OmniAvatar.utils.args_config as args_config
args_config.args = args
# 补充必要的默认参数，确保所有模型组件正常运行
defaults = {
    # 分布式训练参数
    'rank': 0,
    'local_rank': 0,
    'sp_size': 1,
    
    # 训练架构参数
    'train_architecture': 'base',
    'lora_rank': 4,
    'lora_alpha': 4,
    'lora_target_modules': 'q,k,v,o,ffn.0,ffn.2',
    'init_lora_weights': 'kaiming',
    
    # 推理参数
    'max_hw': 720,
    'num_persistent_param_in_dit': 0,
    
    # 模型结构参数（与预训练权重匹配）
    'dim': 5120,
    'eps': 1e-06,
    'ffn_dim': 13824,
    'freq_dim': 256,
    'in_dim': 16,  # 将被 model_config 覆盖为正确值
    'num_heads': 40,
    'num_layers': 40,
    'out_dim': 16,
    'text_len': 512,
    'model_type': 't2v',
    '_class_name': 'WanModel',
    '_diffusers_version': '0.30.0',
}

# 只设置缺失的参数
for k, v in defaults.items():
    if not hasattr(args, k):
        setattr(args, k, v)

# 再次应用 model_config 确保关键参数正确
if hasattr(args, "model_config") and args.model_config:
    for k, v in args.model_config.items():
        setattr(args, k, v)

import gradio as gr
import torch
import tempfile
import os
import math
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as TT
import torch.nn.functional as F
from datetime import datetime
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
from OmniAvatar.utils.io_utils import load_state_dict
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline
from transformers import Wav2Vec2FeatureExtractor
from OmniAvatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg
from OmniAvatar.models.wav2vec import Wav2VecModel
import torchvision.transforms as transforms
import librosa

# ==== 工具函数 ====
def set_seed(seed: int = 42):
    """设置随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def match_size(image_size, h, w):
    """根据输入图片尺寸匹配最适合的预设尺寸"""
    ratio_ = 9999
    size_ = 9999
    select_size = None
    for image_s in image_size:
        ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
        size_tmp = abs(max(image_s) - max(w, h))
        if ratio_tmp < ratio_:
            ratio_ = ratio_tmp
            size_ = size_tmp
            select_size = image_s
        if ratio_ == ratio_tmp:
            if size_ == size_tmp:
                select_size = image_s
    return select_size

def resize_pad(image, ori_size, tgt_size):
    """调整图片尺寸并进行填充，保持长宽比"""
    h, w = ori_size
    scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
    scale_h = int(h * scale_ratio)
    scale_w = int(w * scale_ratio)
    image = transforms.Resize(size=[scale_h, scale_w])(image)
    padding_h = tgt_size[0] - scale_h
    padding_w = tgt_size[1] - scale_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left
    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return image

class WanInferencePipeline(nn.Module):
    """OmniAvatar 推理管道，处理文本、图像、音频输入生成视频"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.rank}")
        
        # 设置数据类型
        if args.dtype=='bf16':
            self.dtype = torch.bfloat16
        elif args.dtype=='fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
            
        # 加载模型
        self.pipe = self.load_model()
        
        # 图像预处理
        if args.i2v:
            chained_trainsforms = []
            chained_trainsforms.append(TT.ToTensor())
            self.transform = TT.Compose(chained_trainsforms)
            
        # 音频编码器
        if args.use_audio:
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    args.wav2vec_path
                )
            self.audio_encoder = Wav2VecModel.from_pretrained(args.wav2vec_path, local_files_only=True).to(device=self.device)
            self.audio_encoder.feature_extractor._freeze_parameters()

    def load_model(self):
        """加载预训练模型和权重"""
        ckpt_path = f'{self.args.exp_path}/pytorch_model.pt'
        assert os.path.exists(ckpt_path), f"pytorch_model.pt not found in {self.args.exp_path}"
        
        if self.args.train_architecture == 'lora':
            self.args.pretrained_lora_path = ckpt_path
        else:
            resume_path = ckpt_path
            
        self.step = 0
        
        # 初始化模型管理器
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models(
            [
                self.args.dit_path.split(","),
                self.args.text_encoder_path,
                self.args.vae_path
            ],
            torch_dtype=self.dtype,
            device='cpu',
        )
        
        # 创建推理管道
        pipe = WanVideoPipeline.from_model_manager(
            model_manager, 
            torch_dtype=self.dtype, 
            device=f"cuda:{self.args.rank}", 
            use_usp=True if self.args.sp_size > 1 else False,
            infer=True
        )
        
        # 加载 LoRA 权重或完整权重
        if self.args.train_architecture == "lora":
            print(f'使用 LoRA: rank={self.args.lora_rank}, alpha={self.args.lora_alpha}')
            self.add_lora_to_model(
                    pipe.denoising_model(),
                    lora_rank=self.args.lora_rank,
                    lora_alpha=self.args.lora_alpha,
                    lora_target_modules=self.args.lora_target_modules,
                    init_lora_weights=self.args.init_lora_weights,
                    pretrained_lora_path=self.args.pretrained_lora_path,
                )
        else:
            missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(load_state_dict(resume_path), strict=True)
            print(f"从 {resume_path} 加载权重，缺失 {len(missing_keys)} 个键，意外 {len(unexpected_keys)} 个键")
            
        pipe.requires_grad_(False)
        pipe.eval()
        pipe.enable_vram_management(num_persistent_param_in_dit=self.args.num_persistent_param_in_dit)
        return pipe
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        """为模型添加 LoRA 适配器并加载权重"""
        from peft import LoraConfig, inject_adapter_in_model
        
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        
        model = inject_adapter_in_model(lora_config, model)
        
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"从 {pretrained_lora_path} 加载了 {num_updated_keys} 个参数，{num_unexpected_keys} 个意外参数")

    def forward(self, prompt, 
                image_path=None, 
                audio_path=None, 
                seq_len=101, 
                height=720, 
                width=720,
                overlap_frame=None,
                num_steps=None,
                negative_prompt=None,
                guidance_scale=None,
                audio_scale=None):
        """执行视频生成推理"""
        overlap_frame = overlap_frame if overlap_frame is not None else self.args.overlap_frame
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale

        if image_path is not None:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
        else:
            image = None
            select_size = [height, width]
        L = int(self.args.max_tokens * 16 * 16 * 4 / select_size[0] / select_size[1])
        L = L // 4 * 4 + 1 if L % 4 != 0 else L - 3
        T = (L + 3) // 4

        if self.args.i2v:
            if self.args.random_prefix_frames:
                fixed_frame = overlap_frame
                assert fixed_frame % 4 == 1
            else:
                fixed_frame = 1
            prefix_lat_frame = (3 + fixed_frame) // 4
            first_fixed_frame = 1
        else:
            fixed_frame = 0
            prefix_lat_frame = 0
            first_fixed_frame = 0

        if audio_path is not None and self.args.use_audio:
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
            input_values = np.squeeze(
                    self.wav_feature_extractor(audio, sampling_rate=16000).input_values
                )
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            ori_audio_len = audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
            input_values = input_values.unsqueeze(0)
            if audio_len < L - first_fixed_frame:
                audio_len = audio_len + ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
            elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
                audio_len = audio_len + ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
            input_values = F.pad(input_values, (0, audio_len * int(self.args.sample_rate / self.args.fps) - input_values.shape[1]), mode='constant', value=0)
            with torch.no_grad():
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            seq_len = audio_len
            audio_embeddings = audio_embeddings.squeeze(0)
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
        else:
            audio_embeddings = None

        times = (seq_len - L + first_fixed_frame) // (L-fixed_frame) + 1
        if times * (L-fixed_frame) + fixed_frame < seq_len:
            times += 1
        video = []
        image_emb = {}
        img_lat = None
        if self.args.i2v:
            self.pipe.load_models_to_device(['vae'])
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
            msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1])
            image_cat = img_lat.repeat(1, 1, T, 1, 1)
            msk[:, :, 1:] = 1
            image_emb["y"] = torch.cat([image_cat, msk], dim=1)
        for t in range(times):
            print(f"[{t+1}/{times}]")
            audio_emb = {}
            if t == 0:
                overlap = first_fixed_frame
            else:
                overlap = fixed_frame
                image_emb["y"][:, -1:, :prefix_lat_frame] = 0
            prefix_overlap = (3 + overlap) // 4
            if audio_embeddings is not None:
                if t == 0:
                    audio_tensor = audio_embeddings[
                            :min(L - overlap, audio_embeddings.shape[0])
                        ]
                else:
                    audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                    audio_tensor = audio_embeddings[
                        audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])
                    ]
                audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                audio_prefix = audio_tensor[-fixed_frame:]
                audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                audio_emb["audio_emb"] = audio_tensor
            else:
                audio_prefix = None
            if image is not None and img_lat is None:
                self.pipe.load_models_to_device(['vae'])
                img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
                assert img_lat.shape[2] == prefix_overlap
            img_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))], dim=2)
            frames, _, latents = self.pipe.log_video(img_lat, prompt, prefix_overlap, image_emb, audio_emb,
                                                 negative_prompt, num_inference_steps=num_steps, 
                                                 cfg_scale=guidance_scale, audio_cfg_scale=audio_scale if audio_scale is not None else guidance_scale,
                                                 return_latent=True,
                                                 tea_cache_l1_thresh=self.args.tea_cache_l1_thresh,tea_cache_model_id="Wan2.1-T2V-14B")
            img_lat = None
            image = (frames[:, -fixed_frame:].clip(0, 1) * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()
            if t == 0:
                video.append(frames)
            else:
                video.append(frames[:, overlap:])
        video = torch.cat(video, dim=1)
        video = video[:, :seq_len + 1]
        return video

# ==== WebUI 界面和推理逻辑 ====

# 设置随机种子
set_seed(args.seed)
inferpipe = None  # 延迟加载模型，仅在首次推理时初始化

def infer(prompt, image, audio):
    """主推理函数，处理用户输入并生成视频"""
    global inferpipe
    
    # 首次调用时初始化模型
    if inferpipe is None:
        print("正在初始化模型...")
        inferpipe = WanInferencePipeline(args)
        print("模型初始化完成")
    
    # 处理图像输入
    image_path = None
    if image is not None:
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp_img.name)
        image_path = tmp_img.name
    
    # 处理音频输入
    audio_path = None
    if audio is not None:
        audio_path = audio  # Gradio Audio 组件直接返回文件路径
    
    # 执行推理
    print(f"开始生成视频，提示词: {prompt}")
    video = inferpipe(
        prompt=prompt,
        image_path=image_path,
        audio_path=audio_path,
        seq_len=args.seq_len
    )
    
    # 保存视频
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    save_video_as_grid_and_mp4(video, os.path.dirname(out_path), args.fps, prompt=prompt, audio_path=audio_path, prefix="result")
    result_video = os.path.join(os.path.dirname(out_path), "result.mp4")
    print(f"视频生成完成: {result_video}")
    
    return result_video

# 创建 Gradio 界面
with gr.Blocks(title="OmniAvatar WebUI") as demo:
    gr.Markdown("# OmniAvatar WebUI")
    gr.Markdown("上传图片、音频和文字提示来生成个性化 Avatar 视频")
    gr.Markdown("💡 **使用说明**: 图片用于定义外观，音频驱动口型和动作，文字描述视频风格")
    
    with gr.Row():
        prompt = gr.Textbox(
            label="文字提示词", 
            placeholder="请输入视频描述，例如：一个人在说话",
            lines=3
        )
    
    with gr.Row():
        with gr.Column():
            image = gr.Image(
                type="pil", 
                label="上传图片（可选）"
            )
        with gr.Column():
            audio = gr.Audio(
                type="filepath", 
                label="上传音频（可选）"
            )
    
    with gr.Row():
        btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
    
    with gr.Row():
        output = gr.Video(label="生成结果", height=400)
    
    # 绑定事件
    btn.click(
        fn=infer, 
        inputs=[prompt, image, audio], 
        outputs=output,
        show_progress=True
    )

# 启动界面

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)