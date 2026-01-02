#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightX2V Qwen-Image 文生图演示程序
基于 Gradio 的独立 Web 界面

作者: AI 技术分享频道
YouTube: https://www.youtube.com/@rongyikanshijie-ai
"""

import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置环境变量
os.environ["PROFILING_DEBUG_LEVEL"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DTYPE"] = "BF16"

import gradio as gr
import torch
from loguru import logger

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config, print_config
from lightx2v.utils.utils import seed_all
from lightx2v.utils.input_info import set_input_info
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401

# 模型路径配置
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MODEL_PATH_ORIGINAL = os.path.join(CHECKPOINT_DIR, "Qwen-Image-2512")
MODEL_PATH_LIGHTNING = os.path.join(CHECKPOINT_DIR, "Qwen-Image-2512-Lightning")
LORA_PATH = os.path.join(MODEL_PATH_LIGHTNING, "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors")

# 配置文件路径
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs", "qwen_image")
CONFIG_ORIGINAL = os.path.join(CONFIG_DIR, "qwen_image_t2i_2512.json")
CONFIG_LORA = os.path.join(CONFIG_DIR, "qwen_image_t2i_2512_lora.json")

# 输出路径
SAVE_DIR = os.path.join(PROJECT_ROOT, "save_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# 示例提示词 - 按功能分类 [功能增强类别, 提示词]
EXAMPLE_PROMPTS = [
    # 人物写实
    [
        "🧑 人物写实 - 精细发丝和自然表情",
        "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.",
    ],
    [
        "🧑 人物写实 - 面部细节和环境背景",
        "A Chinese female college student, around 20 years old, with a very short haircut that conveys a gentle, artistic vibe. Her hair naturally falls to partially cover her cheeks, projecting a tomboyish yet charming demeanor. She has cool-toned fair skin and delicate features, with a slightly shy yet subtly confident expression—her mouth crooked in a playful, youthful smirk. She wears an off-shoulder top, revealing one shoulder, with a well-proportioned figure. The image is framed as a close-up selfie: she dominates the foreground, while the background clearly shows her dormitory—a neatly made bed with white linens on the top bunk, a tidy study desk with organized stationery, and wooden cabinets and drawers. The photo is captured on a smartphone under soft, even ambient lighting, with natural tones, high clarity, and a bright, lively atmosphere full of youthful, everyday energy.",
    ],
    [
        "🧑 人物写实 - 精确姿态语义遵循",
        "An East Asian teenage boy, aged 15–18, with soft, fluffy black short hair and refined facial contours. His large, warm brown eyes sparkle with energy. His fair skin and sunny, open smile convey an approachable, friendly demeanor—no makeup or blemishes. He wears a blue-and-white summer uniform shirt, slightly unbuttoned, made of thin breathable fabric, with black headphones hanging around his neck. His hands are in his pockets, body leaning slightly forward in a relaxed pose, as if engaged in conversation. Behind him lies a summer school playground: lush green grass and a red rubber track in the foreground, blurred school buildings in the distance, a clear blue sky with fluffy white clouds. The bright, airy lighting evokes a joyful, carefree adolescent atmosphere.",
    ],
    [
        "🧑 人物写实 - 年龄特征（皱纹）渲染",
        "An elderly Chinese couple in their 70s in a clean, organized home kitchen. The woman has a kind face and a warm smile, wearing a patterned apron; the man stands behind her, also smiling, as they both gaze at a steaming pot of buns on the stove. The kitchen is bright and tidy, exuding warmth and harmony. The scene is captured with a wide-angle lens to fully show the subjects and their surroundings.",
    ],
    # 自然风景
    [
        "🌿 自然纹理 - 水流植被雾气渲染",
        "A turquoise river winds through a lush canyon. Thick moss and dense ferns blanket the rocky walls; multiple waterfalls cascade from above, enveloped in mist. At noon, sunlight filters through the dense canopy, dappling the river surface with shimmering light. The atmosphere is humid and fresh, pulsing with primal jungle vitality. No humans, text, or artificial traces present.",
    ],
    [
        "🌿 自然纹理 - 海浪与晨雾渲染",
        "At dawn, a thin mist veils the sea. An ancient stone lighthouse stands at the cliff's edge, its beacon faintly visible through the fog. Black rocks are pounded by waves, sending up bursts of white spray. The sky glows in soft blue-purple hues under cool, hazy light—evoking solitude and solemn grandeur.",
    ],
    # 动物毛发
    [
        "🐕 动物毛发 - 精细毛发纹理",
        "An ultra-realistic close-up of a golden retriever outdoors under soft daylight. Hair is exquisitely detailed: strands distinct, color transitioning naturally from warm gold to light cream, light glinting delicately at the tips; a gentle breeze adds subtle volume. Undercoat is soft and dense; guard hairs are long and well-defined, with visible layering. Eyes are moist, expressive; nose is slightly damp with fine specular highlights. Background is softly blurred to emphasize the dog's tangible texture and vivid expression.",
    ],
    [
        "🐕 动物毛发 - 粗糙野生动物纹理",
        "A male argali stands atop a barren, rocky mountainside. Its coarse, dense grey-brown coat covers a powerful, muscular body. Most striking are its massive, thick, outward-spiraling horns—a symbol of wild strength. Its gaze is alert and sharp. The background reveals steep alpine terrain: jagged peaks, sparse low vegetation, and abundant sunlight—conveying the harsh yet majestic wilderness and the animal's resilient vitality.",
    ],
    # 文字渲染
    [
        "📝 文字渲染 - PPT时间轴图文混排",
        '这是一张现代风格的科技感幻灯片，整体采用深蓝色渐变背景。标题是"Qwen-Image发展历程"。下方一条水平延伸的发光时间轴，轴线中间写着"生图路线"。由左侧淡蓝色渐变为右侧深紫色，并以精致的箭头收尾。时间轴上每个节点通过虚线连接至下方醒目的蓝色圆角矩形日期标签，标签内为清晰白色字体，从左向右依次写着："2025年5月6日 Qwen-Image 项目启动""2025年8月4日 Qwen-Image 开源发布""2025年12月31日 Qwen-Image-2512 开源发布" （周围光晕显著）在下方一条水平延伸的发光时间轴，轴线中间写着"编辑路线"。由左侧淡蓝色渐变为右侧深紫色，并以精致的箭头收尾。时间轴上每个节点通过虚线连接至下方醒目的蓝色圆角矩形日期标签，标签内为清晰白色字体，从左向右依次写着："2025年8月18日 Qwen-Image-Edit 开源发布""2025年9月22日 Qwen-Image-Edit-2509 开源发布""2025年12月19日 Qwen-Image-Layered 开源发布""2025年12月23日 Qwen-Image-Edit-2511 开源发布"',
    ],
    [
        "📝 文字渲染 - 产品对比图混合渲染",
        '这是一张现代风格的科技感幻灯片，整体采用深蓝色渐变背景。顶部中央为白色无衬线粗体大字标题"Qwen-Image-2512重磅发布"。画面主体为横向对比图，视觉焦点集中于中间的升级对比区域。左侧为面部光滑没有任何细节的女性人像，质感差；右侧为高度写实的年轻女性肖像，皮肤呈现真实毛孔纹理与细微光影变化，发丝根根分明，眼眸透亮，表情自然，整体质感接近写实摄影。两图像之间以一个绿色流线型箭头链接。造型科技感十足，中部标注"2512质感升级"，使用白色加粗字体，居中显示。箭头两侧有微弱光晕效果，增强动态感。在图像下方，以白色文字呈现三行说明："● 更真实的人物质感。大幅度降低了生成图片的AI感，提升了图像真实性 ● 更细腻的自然纹理。大幅度提升了生成图片的纹理细节。风景图，动物毛发刻画更细腻。● 更复杂的文字渲染。大幅提升了文字渲染的质量。图文混合渲染更准确，排版更好"',
    ],
    [
        "📝 文字渲染 - 工业信息图表复杂布局",
        '这是一幅专业级工业技术信息图表，整体采用深蓝色科技感背景，光线均匀柔和，营造出冷静、精准的现代工业氛围。画面分为左右两大板块，布局清晰，视觉层次分明。左侧板块标题为"实际发生的现象"，以浅蓝色圆角矩形框突出显示，内部排列三个深蓝色按钮式条目，第一个条目展示一堆棕色粉末状原料上滴落水滴的图标，文字为"团聚/结块"，后面配有绿色对钩；第二个条目为一个装有蓝色液体并冒出气泡的锥形瓶，文字为"产生气泡/缺陷"，后面配有绿色对钩；第三个条目为两个生锈的齿轮，文字为"设备腐蚀/催化剂失活"，后面配有绿色对钩。右侧板块标题为"【不会】发生的现象"，使用米黄色圆角矩形框呈现，内部四个条目均置于深灰色背景方框中。图标分别为：一组精密啮合的金属齿轮，文字为"反应效率【显著提高】"，上方覆盖醒目的红色叉号；一捆整齐排列的金属管材，文字为"成品内部【绝对无气泡/孔隙】"，上方覆盖醒目的红色叉号；一条坚固的金属链条正在承受拉力，文字为"材料强度与耐久性【得到增强】"，上方覆盖醒目的红色叉号；一堆腐蚀的扳手，文字为"加工过程【零腐蚀/零副反应风险】"，上方覆盖醒目的红色叉号。底部中央有一行小字注释："注：水分的存在通常会导致负面或干扰性的结果，而非理想或增强的状态"，字体为白色，清晰可读。整体风格现代简约，配色对比强烈，图形符号准确传达技术逻辑，适合用于工业培训或科普演示场景。',
    ],
    [
        "📝 文字渲染 - 网格海报时间标注",
        '这是一幅由十二个分格组成的3×4网格布局的写实摄影作品，整体呈现"健康的一天"主题，画面风格简洁清晰，每一分格独立成景又统一于生活节奏的叙事脉络。第一行分别是"06:00 晨跑唤醒身体"：面部特写，一位女性身穿灰色运动套装，背景是初升的朝阳与葱郁绿树；"06:30 动态拉伸激活关节"：女性身着瑜伽服在阳台做晨间拉伸，身体舒展，背景为淡粉色天空与远山轮廓；"07:30 均衡营养早餐"：桌上摆放全麦面包、牛油果和一杯橙汁，女性微笑着准备用餐；"08:00 补水润燥"：透明玻璃水杯中浮有柠檬片，女性手持水杯轻啜，阳光从左侧斜照入室，杯壁水珠滑落；第二行分别是："09:00 专注高效工作"：女性专注敲击键盘，屏幕显示简洁界面，身旁放有一杯咖啡与一盆绿植；"12:00 静心阅读时光"：女性坐在书桌前翻阅纸质书籍，台灯散发暖光，书页泛黄，旁放半杯红茶；"12:30 午后轻松漫步"：女性在林荫道上漫步，脸部特写；"15:00 茶香伴午后"：女性端着骨瓷茶杯站在窗边，窗外是城市街景与飘动云朵，茶香袅袅；第三行分别是："18:00 运动释放压力"：健身房内，女性正在练习瑜伽；"19:00 美味晚餐"：女性在开放式厨房中切菜，砧板上有番茄与青椒，锅中热气升腾，灯光温暖；"21:00 冥想助眠"：女性盘腿坐在柔软地毯上冥想，双手轻放膝上，闭目宁静；"21:30 进入睡眠"：女性躺在床上休息。整体采用自然光线为主，色调以暖白与米灰为基调，光影层次分明，画面充满温馨的生活气息与规律的节奏感。',
    ],
]

# 模型配置信息
MODEL_CONFIGS = {
    "original": {
        "name": "原始模型 (50步)",
        "description": "Qwen-Image-2512 原始模型，默认50步推理，质量最高但速度较慢",
        "config_json": CONFIG_ORIGINAL,
        "model_path": MODEL_PATH_ORIGINAL,
        "infer_steps": 50,
        "lora_configs": None,
        "available": os.path.exists(MODEL_PATH_ORIGINAL),
    },
    "lora": {
        "name": "蒸馏加速模型 (4步)",
        "description": "使用 LoRA 蒸馏模型加速，仅需4步推理，速度快",
        "config_json": CONFIG_LORA,
        "model_path": MODEL_PATH_ORIGINAL,
        "infer_steps": 4,
        "lora_configs": [{"path": LORA_PATH, "strength": 1.0}],
        "available": os.path.exists(MODEL_PATH_ORIGINAL) and os.path.exists(LORA_PATH),
    },
}

# 全局 Runner 缓存
runners_cache = {}


def get_device_info():
    """获取设备信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
    return "CPU (无 GPU 可用)"


def clear_gpu_memory():
    """清理 GPU 显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_config_for_model(model_type, aspect_ratio="16:9", seed=42):
    """为指定模型类型创建配置"""
    model_info = MODEL_CONFIGS[model_type]
    
    # 读取基础配置
    with open(model_info["config_json"], "r") as f:
        config_data = json.load(f)
    
    # 创建参数对象
    class Args:
        pass
    
    args = Args()
    args.model_cls = "qwen_image"
    args.task = "t2i"
    args.model_path = model_info["model_path"]
    args.config_json = model_info["config_json"]
    args.prompt = ""
    args.negative_prompt = " "
    args.save_result_path = ""
    args.seed = seed
    args.aspect_ratio = aspect_ratio
    args.custom_shape = None
    args.strength = 0.6
    args.image_path = ""
    args.last_frame_path = ""
    args.audio_path = ""
    args.src_ref_images = None
    args.src_video = None
    args.src_mask = None
    args.use_prompt_enhancer = False
    args.return_result_tensor = False
    
    config = set_config(args)
    
    # 更新 LoRA 配置
    if model_info["lora_configs"]:
        config["lora_configs"] = model_info["lora_configs"]
    
    # 使用 torch 实现的 rope，避免 flashinfer 库问题
    config["rope_type"] = "torch"
    
    # 使用 PyTorch SDPA 注意力，避免 flash_attn3 库问题
    config["attn_type"] = "torch_sdpa"
    
    return config


def init_runner(config):
    """初始化 Runner"""
    torch.set_grad_enabled(False)
    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    return runner


def run_single_inference(model_type, prompt, negative_prompt, aspect_ratio, seed, progress_callback=None):
    """运行单个模型推理"""
    global runners_cache
    
    model_info = MODEL_CONFIGS[model_type]
    
    # 检查模型是否可用
    if not model_info.get("available", True):
        note = model_info.get("note", "模型不可用")
        return None, 0, 0, f"模型不可用: {note}"
    
    runner = None
    try:
        # 创建配置
        config = create_config_for_model(model_type, aspect_ratio, seed)
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVE_DIR, f"qwen_image_{model_type}_{timestamp}.png")
        
        # 创建输入参数
        class InputArgs:
            pass
        
        input_args = InputArgs()
        input_args.task = "t2i"
        input_args.prompt = prompt
        input_args.negative_prompt = negative_prompt
        input_args.save_result_path = save_path
        input_args.seed = seed
        input_args.aspect_ratio = aspect_ratio
        input_args.custom_shape = None
        input_args.return_result_tensor = False
        
        # 设置随机种子
        seed_all(seed)
        
        # 初始化 Runner（每次都重新初始化以确保配置正确）
        logger.info(f"初始化 {model_info['name']} Runner...")
        start_init = time.time()
        
        # 清理之前的缓存
        clear_gpu_memory()
        
        runner = init_runner(config)
        init_time = time.time() - start_init
        logger.info(f"Runner 初始化完成，耗时: {init_time:.2f}秒")
        
        # 设置输入信息
        input_info = set_input_info(input_args)
        
        # 运行推理
        logger.info(f"开始推理 {model_info['name']}...")
        start_infer = time.time()
        
        # 手动执行推理步骤，避免 end_run 中的问题
        runner.input_info = input_info
        runner.inputs = runner.run_input_encoder()
        runner.set_target_shape()
        runner.set_img_shapes()
        logger.info(f"input_info: {runner.input_info}")
        latents, generator = runner.run_dit()
        images = runner.run_vae_decoder(latents)
        
        # 保存图像
        image = images[0]
        image.save(save_path)
        logger.info(f"Image saved: {save_path}")
        
        infer_time = time.time() - start_infer
        total_time = time.time() - start_init
        
        logger.info(f"{model_info['name']} 推理完成，推理耗时: {infer_time:.2f}秒，总耗时: {total_time:.2f}秒")
        
        # 清理
        del latents, generator, images
        if runner is not None:
            if hasattr(runner, 'model'):
                del runner.model
            if hasattr(runner, 'vae'):
                del runner.vae
            if hasattr(runner, 'text_encoders'):
                del runner.text_encoders
            del runner
        clear_gpu_memory()
        
        return save_path, infer_time, total_time, None
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"{model_info['name']} 推理失败: {error_msg}")
        if runner is not None:
            try:
                del runner
            except:
                pass
        clear_gpu_memory()
        return None, 0, 0, str(e)


def run_all_models(prompt, negative_prompt, aspect_ratio, seed, progress=gr.Progress()):
    """运行两个模型的推理对比"""
    
    if not prompt or not prompt.strip():
        return None, "", None, "", "❌ 请输入提示词！"
    
    results = {}
    status_messages = []
    
    # 模型推理顺序
    model_order = ["original", "lora"]
    
    for i, model_type in enumerate(model_order):
        model_info = MODEL_CONFIGS[model_type]
        progress((i / 2), f"正在推理: {model_info['name']}...")
        status_messages.append(f"🔄 正在推理: {model_info['name']}...")
        
        image_path, infer_time, total_time, error = run_single_inference(
            model_type, prompt, negative_prompt, aspect_ratio, seed
        )
        
        if error:
            results[model_type] = {
                "image": None,
                "info": f"❌ 推理失败: {error}",
            }
            status_messages.append(f"❌ {model_info['name']} 失败: {error}")
        else:
            results[model_type] = {
                "image": image_path,
                "info": f"✅ 推理耗时: {infer_time:.2f}秒\n📦 总耗时（含加载）: {total_time:.2f}秒",
            }
            status_messages.append(f"✅ {model_info['name']} 完成，耗时: {infer_time:.2f}秒")
    
    progress(1.0, "推理完成！")
    
    # 构建最终状态
    final_status = "📊 推理对比完成！\n\n" + "\n".join(status_messages)
    
    return (
        results["original"]["image"],
        results["original"]["info"],
        results["lora"]["image"],
        results["lora"]["info"],
        final_status,
    )


def run_single_model(model_type, prompt, negative_prompt, aspect_ratio, seed, progress=gr.Progress()):
    """运行单个模型的推理"""
    
    if not prompt or not prompt.strip():
        return None, "❌ 请输入提示词！"
    
    model_info = MODEL_CONFIGS[model_type]
    progress(0.1, f"正在推理: {model_info['name']}...")
    
    image_path, infer_time, total_time, error = run_single_inference(
        model_type, prompt, negative_prompt, aspect_ratio, seed
    )
    
    progress(1.0, "推理完成！")
    
    if error:
        return None, f"❌ 推理失败: {error}"
    
    return image_path, f"✅ 推理耗时: {infer_time:.2f}秒\n📦 总耗时（含加载）: {total_time:.2f}秒"


def update_prompt_from_example(example_category):
    """从示例中更新提示词"""
    for category, prompt in EXAMPLE_PROMPTS:
        if category == example_category:
            return prompt
    return ""


def create_ui():
    """创建 Gradio 界面"""
    
    # 检查模型是否存在
    model_status = []
    if os.path.exists(MODEL_PATH_ORIGINAL):
        model_status.append("✅ Qwen-Image-2512 原始模型已就绪")
        MODEL_CONFIGS["original"]["available"] = True
    else:
        model_status.append("❌ Qwen-Image-2512 原始模型未找到")
        MODEL_CONFIGS["original"]["available"] = False
    
    if os.path.exists(LORA_PATH):
        model_status.append("✅ Lightning LoRA 模型已就绪")
        MODEL_CONFIGS["lora"]["available"] = os.path.exists(MODEL_PATH_ORIGINAL)
    else:
        model_status.append("⚠️ Lightning LoRA 模型未找到（蒸馏加速模型将不可用）")
        MODEL_CONFIGS["lora"]["available"] = False
    
    model_status_text = "\n".join(model_status)
    
    # 创建示例选项
    example_choices = [category for category, _ in EXAMPLE_PROMPTS]
    
    with gr.Blocks(
        title="LightX2V Qwen-Image 文生图演示",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 20px; }
        .footer { text-align: center; margin-top: 20px; padding: 10px; background: #f0f0f0; border-radius: 8px; }
        .model-card { border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin: 5px; }
        """
    ) as demo:
        
        # YouTube 频道信息 - 放在顶部，使用素色调
        gr.HTML(
            """
            <div style="text-align: center; padding: 10px 15px; background: #fafafa; border-radius: 6px; margin-bottom: 12px; border: 1px solid #eee;">
                <span style="color: #666; font-size: 14px;">📺 <b>AI 技术分享频道</b></span>
                <span style="color: #999; font-size: 12px; margin-left: 12px;">欢迎订阅我的 YouTube 频道，获取更多 AI 技术教程！</span>
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" 
                   style="display: inline-block; margin-left: 12px; padding: 4px 12px; background: #777; color: white; 
                          text-decoration: none; border-radius: 3px; font-size: 12px;">
                    🔔 点击订阅
                </a>
            </div>
            """
        )
        
        # 标题区域
        gr.Markdown(
            """
            # 🎨 LightX2V Qwen-Image 文生图演示
            
            基于 **Qwen-Image-2512** 模型的文本生成图像演示程序
            
            ---
            """
        )
        
        # 设备和模型状态信息
        with gr.Row():
            gr.Markdown(f"**🖥️ 设备信息**: {get_device_info()}")
        
        with gr.Accordion("📦 模型状态", open=False):
            gr.Markdown(model_status_text)
        
        # 主界面
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入参数")
                
                # 示例选择
                example_dropdown = gr.Dropdown(
                    label="🎯 示例提示词（选择后自动填充）",
                    choices=example_choices,
                    value=None,
                    interactive=True,
                )
                
                # 提示词输入
                prompt_input = gr.Textbox(
                    label="✍️ 提示词 (Prompt)",
                    placeholder="请输入图像描述...",
                    lines=8,
                    max_lines=15,
                )
                
                negative_prompt_input = gr.Textbox(
                    label="🚫 负面提示词 (Negative Prompt)",
                    value=" ",
                    placeholder="输入不希望出现的内容...",
                    lines=2,
                )
                
                # 参数设置
                with gr.Row():
                    aspect_ratio = gr.Dropdown(
                        label="📐 宽高比",
                        choices=["16:9", "9:16", "1:1", "4:3", "3:4"],
                        value="16:9",
                    )
                    seed_input = gr.Number(
                        label="🎲 随机种子",
                        value=42,
                        precision=0,
                    )
                
                # 运行按钮
                run_all_btn = gr.Button("🚀 运行全部模型对比", variant="primary", size="lg")
                
                with gr.Row():
                    run_original_btn = gr.Button("▶️ 仅运行原始模型", size="sm")
                    run_lora_btn = gr.Button("▶️ 仅运行蒸馏模型", size="sm")
                
                # 状态显示
                status_output = gr.Textbox(
                    label="📊 运行状态",
                    lines=6,
                    interactive=False,
                )
        
        # 右侧：输出区域
        gr.Markdown("### 🖼️ 生成结果对比")
        
        with gr.Row():
            # 原始模型输出
            with gr.Column(scale=1):
                gr.Markdown("#### 原始模型 (50步)")
                gr.Markdown("*质量最高，速度较慢*")
                output_original = gr.Image(
                    label="原始模型输出",
                    type="filepath",
                    height=400,
                )
                info_original = gr.Textbox(
                    label="⏱️ 耗时信息",
                    lines=2,
                    interactive=False,
                )
            
            # 蒸馏模型输出
            with gr.Column(scale=1):
                gr.Markdown("#### 蒸馏加速模型 (4步)")
                gr.Markdown("*LoRA蒸馏，速度快*")
                output_lora = gr.Image(
                    label="蒸馏模型输出",
                    type="filepath",
                    height=400,
                )
                info_lora = gr.Textbox(
                    label="⏱️ 耗时信息",
                    lines=2,
                    interactive=False,
                )
        
        # 底部信息
        gr.Markdown(
            """
            ---
            
            ### 📚 使用说明
            
            1. **选择示例**：从下拉菜单选择预设提示词，或直接输入自定义提示词
            2. **调整参数**：设置宽高比和随机种子
            3. **运行推理**：
               - 点击"运行全部模型对比"同时运行两个模型进行对比
               - 或点击单独按钮仅运行特定模型
            4. **查看结果**：比较不同模型的生成质量和推理速度
            
            ### ⚡ 模型说明
            
            | 模型 | 推理步数 | 特点 |
            |------|----------|------|
            | 原始模型 | 50步 | 质量最高，但速度较慢 |
            | 蒸馏模型 | 4步 | 使用LoRA加速，速度快，质量略降 |
            
            ### 📦 模型下载地址
            
            - **原始模型**: [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)
            - **Lightning 加速模型**: [lightx2v/Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning)
            
            ---
            """
        )
        
        # 底部 YouTube 频道信息 - 素色调
        gr.HTML(
            """
            <div style="text-align: center; padding: 10px; background: #fafafa; border-radius: 6px; margin: 10px 0; border: 1px solid #eee;">
                <span style="color: #666; font-size: 13px;">📺 <b>AI 技术分享频道</b></span>
                <span style="color: #999; font-size: 11px; margin-left: 8px;">欢迎订阅获取更多 AI 技术教程！</span>
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" 
                   style="display: inline-block; margin-left: 8px; padding: 3px 10px; background: #777; color: white; 
                          text-decoration: none; border-radius: 3px; font-size: 11px;">
                    🔔 点击订阅
                </a>
            </div>
            """
        )
        
        gr.Markdown(
            """
            <div style="text-align: center; color: #aaa; font-size: 11px; margin-top: 15px;">
                Powered by LightX2V | Qwen-Image-2512 | 
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="color: #aaa;">AI 技术分享频道</a>
            </div>
            """
        )
        
        # 事件绑定
        example_dropdown.change(
            fn=update_prompt_from_example,
            inputs=[example_dropdown],
            outputs=[prompt_input],
        )
        
        run_all_btn.click(
            fn=run_all_models,
            inputs=[prompt_input, negative_prompt_input, aspect_ratio, seed_input],
            outputs=[
                output_original, info_original,
                output_lora, info_lora,
                status_output,
            ],
        )
        
        run_original_btn.click(
            fn=lambda p, n, a, s: run_single_model("original", p, n, a, s),
            inputs=[prompt_input, negative_prompt_input, aspect_ratio, seed_input],
            outputs=[output_original, info_original],
        )
        
        run_lora_btn.click(
            fn=lambda p, n, a, s: run_single_model("lora", p, n, a, s),
            inputs=[prompt_input, negative_prompt_input, aspect_ratio, seed_input],
            outputs=[output_lora, info_lora],
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightX2V Qwen-Image 文生图演示")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    args = parser.parse_args()
    
    logger.info("正在启动 LightX2V Qwen-Image 文生图演示...")
    logger.info(f"设备信息: {get_device_info()}")
    logger.info(f"项目路径: {PROJECT_ROOT}")
    logger.info(f"模型路径: {CHECKPOINT_DIR}")
    
    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
