#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M6-5-新模型测试.py - 微调模型与原始模型对比测试

功能：
1. 加载微调后的custom.pt模型和原始的large-v3.pt模型
2. 对同一音频文件进行推理
3. 对比两个模型的识别结果
4. 评估准确率和相似度
5. 支持批量测试多个音频文件

使用方法：
    # 直接运行测试（无需命令行参数）
    python M6-5-新模型测试.py
    
    # 可在main()函数中修改以下配置：
    # - test_mode: "single"(单文件) 或 "batch"(批量测试)
    # - audio_path: 单个音频文件路径
    # - audio_dir: 音频目录路径
    # - custom_model_path: 微调模型路径
    # - original_model_path: 原始模型路径

环境要求：
    pip install openai-whisper jiwer

作者：GitHub Copilot
日期：2025-07-07
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import whisper
import torch
import numpy as np
from jiwer import wer, cer


def load_single_model(model_path: str, model_name: str) -> whisper.Whisper:
    """
    加载单个模型
    
    Args:
        model_path: 模型路径
        model_name: 模型名称（用于日志）
        
    Returns:
        whisper.Whisper: 加载的模型
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ {model_name}不存在: {model_path}")
    
    # 加载模型
    print(f"📂 正在加载{model_name}: {model_path}")
    start_time = time.time()
    model = whisper.load_model(model_path)
    load_time = time.time() - start_time
    print(f"✅ {model_name}加载完成 ({load_time:.2f}s)")
    
    return model


def release_model(model: whisper.Whisper, model_name: str) -> None:
    """
    释放模型资源
    
    Args:
        model: 要释放的模型
        model_name: 模型名称（用于日志）
    """
    try:
        # 将模型移到CPU并删除引用
        if hasattr(model, 'device'):
            model.to('cpu')
        del model
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"🧹 {model_name}资源已释放")
    except Exception as e:
        print(f"⚠️  释放{model_name}资源时出错: {e}")


def transcribe_audio(model: whisper.Whisper, audio_path: str, model_name: str) -> Dict:
    """
    使用指定模型对音频进行转录
    
    Args:
        model: Whisper模型
        audio_path: 音频文件路径
        model_name: 模型名称（用于日志）
        
    Returns:
        dict: 转录结果
    """
    print(f"🎵 使用{model_name}转录: {os.path.basename(audio_path)}")
    
    start_time = time.time()
    
    # 进行转录（使用优化后的参数，解决截断问题）
    result = model.transcribe(
        audio_path,
        language="zh",  # 中文
        task="transcribe",
        temperature=0.3,  # 适度随机性，避免过早截断
        beam_size=1,      # 贪婪搜索
        best_of=1,        # 简化搜索
        patience=1.0,     # 耐心参数
        length_penalty=0.0,  # 中性长度惩罚
        suppress_tokens=[],  # 不抑制任何token
        initial_prompt="",   # 空初始提示
        condition_on_previous_text=False,  # 不依赖前文，避免截断
        fp16=torch.cuda.is_available(),   # 使用半精度（如果GPU可用）
        compression_ratio_threshold=2.4,  # 压缩比阈值
        logprob_threshold=-1.0,           # 对数概率阈值
        no_speech_threshold=0.6,          # 无语音阈值
    )
    
    inference_time = time.time() - start_time
    
    # 提取关键信息
    transcription = {
        'text': result['text'].strip(),
        'language': result['language'],
        'segments': result['segments'],
        'inference_time': inference_time,
        'model_name': model_name
    }
    
    print(f"⏱️  {model_name}推理耗时: {inference_time:.2f}s")
    print(f"📝 {model_name}识别结果: {transcription['text']}")
    
    return transcription


def calculate_similarity_metrics(text1: str, text2: str) -> Dict[str, float]:
    """
    计算两个文本之间的相似度指标
    
    Args:
        text1: 文本1
        text2: 文本2
        
    Returns:
        dict: 相似度指标
    """
    if not text1.strip() or not text2.strip():
        return {
            'wer': 1.0,  # 词错误率100%
            'cer': 1.0,  # 字符错误率100%
            'similarity': 0.0  # 相似度0%
        }
    
    try:
        # 计算词错误率 (Word Error Rate)
        word_error_rate = wer(text1, text2)
        
        # 计算字符错误率 (Character Error Rate)
        char_error_rate = cer(text1, text2)
        
        # 简单的相似度计算（基于字符匹配）
        common_chars = set(text1.lower()) & set(text2.lower())
        total_chars = set(text1.lower()) | set(text2.lower())
        similarity = len(common_chars) / len(total_chars) if total_chars else 0.0
        
        return {
            'wer': word_error_rate,
            'cer': char_error_rate,
            'similarity': similarity
        }
    except Exception as e:
        print(f"⚠️  计算相似度时出错: {e}")
        return {
            'wer': 1.0,
            'cer': 1.0,
            'similarity': 0.0
        }


def compare_results(custom_result: Dict, original_result: Dict, audio_path: str) -> Dict:
    """
    对比两个模型的推理结果
    
    Args:
        custom_result: 微调模型结果
        original_result: 原始模型结果
        audio_path: 音频文件路径
        
    Returns:
        dict: 对比结果
    """
    print(f"\n" + "="*80)
    print(f"📊 音频文件对比结果: {os.path.basename(audio_path)}")
    print(f"="*80)
    
    custom_text = custom_result['text']
    original_text = original_result['text']
    
    # 打印识别结果
    print(f"🎯 微调模型结果: {custom_text}")
    print(f"🔄 原始模型结果: {original_text}")
    
    # 计算相似度指标
    metrics = calculate_similarity_metrics(custom_text, original_text)
    
    # 打印性能对比
    print(f"\n📈 性能对比:")
    print(f"   微调模型推理时间: {custom_result['inference_time']:.2f}s")
    print(f"   原始模型推理时间: {original_result['inference_time']:.2f}s")
    
    speed_improvement = (original_result['inference_time'] - custom_result['inference_time']) / original_result['inference_time'] * 100
    if speed_improvement > 0:
        print(f"   ⚡ 微调模型速度提升: {speed_improvement:.1f}%")
    else:
        print(f"   ⏳ 微调模型速度降低: {abs(speed_improvement):.1f}%")
    
    # 打印相似度指标
    print(f"\n🔍 文本对比指标:")
    print(f"   词错误率 (WER): {metrics['wer']:.3f}")
    print(f"   字符错误率 (CER): {metrics['cer']:.3f}")
    print(f"   文本相似度: {metrics['similarity']:.3f}")
    
    # 打印段落级别信息
    print(f"\n📝 段落数量对比:")
    print(f"   微调模型段落数: {len(custom_result['segments'])}")
    print(f"   原始模型段落数: {len(original_result['segments'])}")
    
    comparison = {
        'audio_file': os.path.basename(audio_path),
        'custom_text': custom_text,
        'original_text': original_text,
        'custom_time': custom_result['inference_time'],
        'original_time': original_result['inference_time'],
        'speed_improvement': speed_improvement,
        'metrics': metrics,
        'custom_segments': len(custom_result['segments']),
        'original_segments': len(original_result['segments'])
    }
    
    return comparison


def test_single_audio_optimized(custom_model_path: str, original_model_path: str, 
                               audio_path: str) -> Dict:
    """
    优化版本：逐个加载模型进行测试，节省内存
    
    Args:
        custom_model_path: 微调模型路径
        original_model_path: 原始模型路径
        audio_path: 音频文件路径
        
    Returns:
        dict: 对比结果
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ 音频文件不存在: {audio_path}")
    
    print(f"\n🎵 开始测试音频: {audio_path}")
    print("💡 采用逐个加载策略，节省显存占用...")
    
    # 第一步：加载微调模型并推理
    print(f"\n{'='*50} 第1步：微调模型推理 {'='*50}")
    custom_model = load_single_model(custom_model_path, "微调模型")
    custom_result = transcribe_audio(custom_model, audio_path, "微调模型")
    release_model(custom_model, "微调模型")
    
    # 第二步：加载原始模型并推理
    print(f"\n{'='*50} 第2步：原始模型推理 {'='*50}")
    original_model = load_single_model(original_model_path, "原始模型")
    original_result = transcribe_audio(original_model, audio_path, "原始模型")
    release_model(original_model, "原始模型")
    
    # 第三步：对比结果
    print(f"\n{'='*50} 第3步：结果对比分析 {'='*50}")
    comparison = compare_results(custom_result, original_result, audio_path)
    
    return comparison


def test_audio_directory_optimized(custom_model_path: str, original_model_path: str,
                                 audio_dir: str) -> List[Dict]:
    """
    优化版本：逐个加载模型测试音频目录
    
    Args:
        custom_model_path: 微调模型路径  
        original_model_path: 原始模型路径
        audio_dir: 音频目录路径
        
    Returns:
        list: 所有对比结果
    """
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"❌ 音频目录不存在: {audio_dir}")
    
    # 支持的音频格式
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'}
    
    # 查找所有音频文件
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
        audio_files.extend(Path(audio_dir).glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"⚠️  在 {audio_dir} 目录下未找到音频文件")
        return []
    
    print(f"📁 在 {audio_dir} 目录下找到 {len(audio_files)} 个音频文件")
    print("💡 采用逐个加载策略，每个文件独立测试...")
    
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*20} 测试进度: {i}/{len(audio_files)} {'='*20}")
        
        try:
            result = test_single_audio_optimized(custom_model_path, original_model_path, str(audio_file))
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {audio_file} 时出错: {e}")
            continue
    
    return results


def print_summary(results: List[Dict]) -> None:
    """
    打印测试总结
    
    Args:
        results: 所有测试结果
    """
    if not results:
        print("❌ 没有可用的测试结果")
        return
    
    print(f"\n" + "="*80)
    print(f"📊 测试总结 (共 {len(results)} 个音频文件)")
    print(f"="*80)
    
    # 计算平均指标
    avg_wer = np.mean([r['metrics']['wer'] for r in results])
    avg_cer = np.mean([r['metrics']['cer'] for r in results])
    avg_similarity = np.mean([r['metrics']['similarity'] for r in results])
    avg_speed_improvement = np.mean([r['speed_improvement'] for r in results])
    
    total_custom_time = sum([r['custom_time'] for r in results])
    total_original_time = sum([r['original_time'] for r in results])
    
    print(f"📈 平均指标:")
    print(f"   平均词错误率 (WER): {avg_wer:.3f}")
    print(f"   平均字符错误率 (CER): {avg_cer:.3f}")
    print(f"   平均文本相似度: {avg_similarity:.3f}")
    print(f"   平均速度提升: {avg_speed_improvement:.1f}%")
    
    print(f"\n⏱️  总推理时间:")
    print(f"   微调模型总时间: {total_custom_time:.2f}s")
    print(f"   原始模型总时间: {total_original_time:.2f}s")
    print(f"   总时间节省: {total_original_time - total_custom_time:.2f}s")
    
    # 找出最佳和最差结果
    best_similarity = max(results, key=lambda x: x['metrics']['similarity'])
    worst_similarity = min(results, key=lambda x: x['metrics']['similarity'])
    
    print(f"\n🏆 最佳一致性: {best_similarity['audio_file']} (相似度: {best_similarity['metrics']['similarity']:.3f})")
    print(f"💡 最大差异: {worst_similarity['audio_file']} (相似度: {worst_similarity['metrics']['similarity']:.3f})")


def test_checkpoint_vs_custom(checkpoint_path: str, custom_model_path: str, audio_path: str) -> Dict:
    """
    专门对比checkpoint-4000和custom.pt的差异
    
    Args:
        checkpoint_path: HuggingFace checkpoint路径
        custom_model_path: 转换后的OpenAI格式模型路径
        audio_path: 音频文件路径
        
    Returns:
        dict: 对比结果
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import librosa
    
    print(f"\n🔍 开始对比 checkpoint-4000 vs custom.pt")
    print(f"📂 HF Checkpoint: {checkpoint_path}")
    print(f"📂 OpenAI Model: {custom_model_path}")
    print(f"🎵 测试音频: {audio_path}")
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 第一步：使用HuggingFace checkpoint
    print(f"\n{'='*50} 第1步：HF Checkpoint推理 {'='*50}")
    hf_model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    hf_processor = WhisperProcessor.from_pretrained(checkpoint_path)
    
    # HF模型推理
    input_features = hf_processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    
    # 使用更保守的参数避免长度错误
    start_time = time.time()
    generated_ids = hf_model.generate(
        input_features,
        language="zh",
        task="transcribe", 
        max_new_tokens=400,  # 减少到400避免超出限制
        num_beams=5,
        temperature=0.0,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )
    hf_time = time.time() - start_time
    
    hf_text = hf_processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0].strip()
    print(f"✅ HF Checkpoint结果: {hf_text}")
    print(f"⏱️  HF推理耗时: {hf_time:.2f}s")
    
    # 释放HF模型
    del hf_model, hf_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 第二步：使用OpenAI格式模型
    print(f"\n{'='*50} 第2步：OpenAI Model推理 {'='*50}")
    openai_model = load_single_model(custom_model_path, "OpenAI模型")
    
    start_time = time.time()
    # 使用优化的参数解决截断问题
    openai_result = openai_model.transcribe(
        audio_path,
        language="zh",
        task="transcribe",
        temperature=0.3,  # 增加随机性，避免截断
        beam_size=1,      # 贪婪搜索
        best_of=1,
        patience=1.0,
        length_penalty=0.0,  # 中性长度惩罚
        suppress_tokens=[],  # 不抑制任何token
        initial_prompt="",   # 空初始提示
        condition_on_previous_text=False,  # 不依赖前文，关键！
        fp16=torch.cuda.is_available(),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    openai_time = time.time() - start_time
    
    openai_text = openai_result['text'].strip()
    print(f"✅ OpenAI Model结果: {openai_text}")
    print(f"⏱️  OpenAI推理耗时: {openai_time:.2f}s")
    
    release_model(openai_model, "OpenAI模型")
    
    # 第三步：详细对比分析
    print(f"\n{'='*50} 第3步：详细对比分析 {'='*50}")
    
    # 计算相似度指标
    metrics = calculate_similarity_metrics(hf_text, openai_text)
    
    print(f"🔍 文本对比:")
    print(f"   HF Checkpoint: '{hf_text}'")
    print(f"   OpenAI Model:  '{openai_text}'")
    print(f"   文本长度差异: {len(hf_text)} vs {len(openai_text)} 字符")
    
    print(f"\n📊 相似度指标:")
    print(f"   词错误率 (WER): {metrics['wer']:.3f}")
    print(f"   字符错误率 (CER): {metrics['cer']:.3f}")
    print(f"   文本相似度: {metrics['similarity']:.3f}")
    
    print(f"\n⏱️  性能对比:")
    print(f"   HF推理时间: {hf_time:.2f}s")
    print(f"   OpenAI推理时间: {openai_time:.2f}s")
    
    if metrics['wer'] > 0.1 or metrics['cer'] > 0.1:
        print(f"\n⚠️  检测到显著差异！")
        print(f"   这可能表明模型转换过程中存在问题")
        print(f"   建议检查转换脚本或重新转换模型")
    else:
        print(f"\n✅ 模型转换质量良好，差异在可接受范围内")
    
    return {
        'hf_text': hf_text,
        'openai_text': openai_text,
        'hf_time': hf_time,
        'openai_time': openai_time,
        'metrics': metrics,
        'audio_file': os.path.basename(audio_path)
    }


def main():
    """主函数"""
    # ===== 直接在代码中配置测试参数 =====
    # 模型路径配置
    custom_model_path = '../models/custom.pt'      # 微调模型路径
    original_model_path = '../models/large-v3.pt'  # 原始模型路径
    checkpoint_path = 'whisper-large-v3-finetuned/checkpoint-4000'  # HF checkpoint路径
    
    # 测试模式配置
    test_mode = "single"  # "single", "batch", "checkpoint_compare"
    
    if test_mode == "checkpoint_compare":
        # checkpoint vs custom.pt 对比模式
        audio_path = "../audio/00000034.wav"
        audio_dir = None
    elif test_mode == "single":
        # 单文件测试模式
        audio_path = "../audio/00000034.wav"  # 指定要测试的音频文件
        audio_dir = None
    else:
        # 批量测试模式
        audio_path = None
        audio_dir = "../audio"  # 指定要测试的音频目录
    
    print("🚀 开始微调模型与原始模型对比测试")
    
    if test_mode == "checkpoint_compare":
        print(f"🔍 测试模式: Checkpoint vs Custom.pt 对比")
        print(f"⚙️  HF Checkpoint: {checkpoint_path}")
        print(f"⚙️  OpenAI Model: {custom_model_path}")
        print(f"🎵 音频文件: {audio_path}")
    elif test_mode == "single":
        print(f"🎵 测试模式: 单文件测试")
        print(f"⚙️  微调模型路径: {custom_model_path}")
        print(f"⚙️  原始模型路径: {original_model_path}")
        print(f"📁 音频文件: {audio_path}")
    else:
        print(f"🎵 测试模式: 批量测试")
        print(f"⚙️  微调模型路径: {custom_model_path}")
        print(f"⚙️  原始模型路径: {original_model_path}")
        print(f"📁 音频目录: {audio_dir}")
    
    try:
        # 执行测试 - 使用优化的逐个加载策略
        if test_mode == "checkpoint_compare":
            # checkpoint vs custom.pt 专项对比
            result = test_checkpoint_vs_custom(checkpoint_path, custom_model_path, audio_path)
            print(f"\n{'='*80}")
            print(f"📊 对比测试完成")
            print(f"{'='*80}")
        elif test_mode == "single":
            # 单个音频文件测试
            result = test_single_audio_optimized(custom_model_path, original_model_path, audio_path)
            print_summary([result])
        else:
            # 批量音频测试
            results = test_audio_directory_optimized(custom_model_path, original_model_path, audio_dir)
            print_summary(results)
        
        print(f"\n✅ 测试完成！")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
