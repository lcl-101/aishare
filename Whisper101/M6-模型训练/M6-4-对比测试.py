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


def compare_results_with_truth(custom_result: Dict, original_result: Dict, audio_path: str, 
                              ground_truth: str = None) -> Dict:
    """
    对比两个模型的推理结果，包含真实标签评估
    
    Args:
        custom_result: 微调模型结果
        original_result: 原始模型结果
        audio_path: 音频文件路径
        ground_truth: 真实标签文本
        
    Returns:
        dict: 对比结果
    """
    print(f"\n" + "="*80)
    print(f"📊 音频文件对比结果: {os.path.basename(audio_path)}")
    print(f"="*80)
    
    custom_text = custom_result['text']
    original_text = original_result['text']
    
    # 尝试自动加载真实标签
    if ground_truth is None:
        ground_truth = load_ground_truth(audio_path)
    
    # 打印识别结果
    if ground_truth:
        print(f"✅ 真实标签: {ground_truth}")
        print(f"🎯 微调模型结果: {custom_text}")
        print(f"🔄 原始模型结果: {original_text}")
    else:
        print(f"🎯 微调模型结果: {custom_text}")
        print(f"🔄 原始模型结果: {original_text}")
        print(f"⚠️ 未找到真实标签文件")
    
    # 计算相对于真实标签的准确性
    custom_accuracy = None
    original_accuracy = None
    
    if ground_truth:
        print(f"\n🎯 相对于真实标签的准确性评估:")
        
        custom_accuracy = calculate_accuracy_metrics(custom_text, ground_truth)
        original_accuracy = calculate_accuracy_metrics(original_text, ground_truth)
        
        print(f"   微调模型准确率: {custom_accuracy['accuracy']:.1%}")
        print(f"   微调模型WER: {custom_accuracy['wer']:.3f}")
        print(f"   微调模型精确匹配: {'✅ 是' if custom_accuracy['exact_match'] else '❌ 否'}")
        
        print(f"   原始模型准确率: {original_accuracy['accuracy']:.1%}")
        print(f"   原始模型WER: {original_accuracy['wer']:.3f}")
        print(f"   原始模型精确匹配: {'✅ 是' if original_accuracy['exact_match'] else '❌ 否'}")
        
        # 判断哪个模型更准确
        if custom_accuracy['accuracy'] > original_accuracy['accuracy']:
            improvement = (custom_accuracy['accuracy'] - original_accuracy['accuracy'])
            print(f"\n🏆 微调模型更准确 (准确率高 {improvement:.1%})")
            
            # 特别分析：如果微调模型完全正确而原始模型错误
            if custom_accuracy['exact_match'] and not original_accuracy['exact_match']:
                print(f"✨ 重要发现: 微调模型实现精确匹配，而原始模型识别错误")
                print(f"📈 这表明微调训练显著提升了模型在此类音频上的准确性")
                
        elif original_accuracy['accuracy'] > custom_accuracy['accuracy']:
            degradation = (original_accuracy['accuracy'] - custom_accuracy['accuracy'])
            print(f"\n⚠️ 原始模型更准确 (准确率高 {degradation:.1%})")
            print(f"🔍 建议: 微调模型在此样本上表现不佳，可能需要检查训练数据")
        else:
            print(f"\n🤝 两个模型准确率相同")
        
        # 分析推理速度差异的合理性
        print(f"\n⚡ 推理速度分析:")
        speed_ratio = custom_result['inference_time'] / original_result['inference_time']
        if speed_ratio > 1.5:
            print(f"⚠️ 微调模型推理较慢 ({speed_ratio:.1f}x)，可能原因:")
            print(f"   • 模型权重变化导致计算复杂度增加")
            print(f"   • 转换过程中的精度变化")
            print(f"   • 内存访问模式的差异")
            print(f"💡 建议: 如果准确率提升显著，适度的速度降低是可接受的")
        elif speed_ratio > 1.2:
            print(f"📊 微调模型推理稍慢 ({speed_ratio:.1f}x)，属于正常范围")
        else:
            print(f"✅ 推理速度表现良好 ({speed_ratio:.1f}x)")
    
    # 计算模型间相似度指标（用于了解两个模型的差异程度）
    inter_model_metrics = calculate_similarity_metrics(custom_text, original_text)
    
    # 打印性能对比
    print(f"\n📈 性能对比:")
    print(f"   微调模型推理时间: {custom_result['inference_time']:.2f}s")
    print(f"   原始模型推理时间: {original_result['inference_time']:.2f}s")
    
    speed_improvement = (original_result['inference_time'] - custom_result['inference_time']) / original_result['inference_time'] * 100
    if speed_improvement > 0:
        print(f"   ⚡ 微调模型速度提升: {speed_improvement:.1f}%")
    else:
        print(f"   ⏳ 微调模型速度降低: {abs(speed_improvement):.1f}%")
    
    # 打印模型间差异指标
    print(f"\n🔍 模型间差异指标:")
    print(f"   词错误率 (WER): {inter_model_metrics['wer']:.3f}")
    print(f"   字符错误率 (CER): {inter_model_metrics['cer']:.3f}")
    print(f"   文本相似度: {inter_model_metrics['similarity']:.3f}")
    
    # 打印段落级别信息
    print(f"\n📝 段落数量对比:")
    print(f"   微调模型段落数: {len(custom_result['segments'])}")
    print(f"   原始模型段落数: {len(original_result['segments'])}")
    
    comparison = {
        'audio_file': os.path.basename(audio_path),
        'ground_truth': ground_truth,
        'custom_text': custom_text,
        'original_text': original_text,
        'custom_time': custom_result['inference_time'],
        'original_time': original_result['inference_time'],
        'speed_improvement': speed_improvement,
        'inter_model_metrics': inter_model_metrics,
        'custom_accuracy': custom_accuracy,
        'original_accuracy': original_accuracy,
        'custom_segments': len(custom_result['segments']),
        'original_segments': len(original_result['segments'])
    }
    
    return comparison


def compare_results(custom_result: Dict, original_result: Dict, audio_path: str) -> Dict:
    """
    向后兼容的对比函数
    """
    return compare_results_with_truth(custom_result, original_result, audio_path)


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
    comparison = compare_results_with_truth(custom_result, original_result, audio_path)
    
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


def print_summary_with_truth(results: List[Dict]) -> None:
    """
    打印包含真实标签评估的测试总结
    
    Args:
        results: 所有测试结果
    """
    if not results:
        print("❌ 没有可用的测试结果")
        return
    
    print(f"\n" + "="*80)
    print(f"📊 测试总结 (共 {len(results)} 个音频文件)")
    print(f"="*80)
    
    # 计算模型间比较指标
    inter_model_wer = np.mean([r['inter_model_metrics']['wer'] for r in results])
    inter_model_cer = np.mean([r['inter_model_metrics']['cer'] for r in results])
    inter_model_similarity = np.mean([r['inter_model_metrics']['similarity'] for r in results])
    avg_speed_improvement = np.mean([r['speed_improvement'] for r in results])
    
    total_custom_time = sum([r['custom_time'] for r in results])
    total_original_time = sum([r['original_time'] for r in results])
    
    print(f"🔍 模型间差异指标:")
    print(f"   平均词错误率 (WER): {inter_model_wer:.3f}")
    print(f"   平均字符错误率 (CER): {inter_model_cer:.3f}")
    print(f"   平均文本相似度: {inter_model_similarity:.3f}")
    print(f"   平均速度提升: {avg_speed_improvement:.1f}%")
    
    print(f"\n⏱️ 总推理时间:")
    print(f"   微调模型总时间: {total_custom_time:.2f}s")
    print(f"   原始模型总时间: {total_original_time:.2f}s")
    print(f"   总时间节省: {total_original_time - total_custom_time:.2f}s")
    
    # 分析有真实标签的结果
    truth_results = [r for r in results if r.get('ground_truth') and r.get('custom_accuracy') and r.get('original_accuracy')]
    
    if truth_results:
        print(f"\n🎯 真实标签准确性分析 (共 {len(truth_results)} 个有标签文件):")
        
        custom_accuracies = [r['custom_accuracy']['accuracy'] for r in truth_results]
        original_accuracies = [r['original_accuracy']['accuracy'] for r in truth_results]
        
        avg_custom_accuracy = np.mean(custom_accuracies)
        avg_original_accuracy = np.mean(original_accuracies)
        
        custom_exact_matches = sum([1 for r in truth_results if r['custom_accuracy']['exact_match']])
        original_exact_matches = sum([1 for r in truth_results if r['original_accuracy']['exact_match']])
        
        print(f"   微调模型平均准确率: {avg_custom_accuracy:.1%}")
        print(f"   原始模型平均准确率: {avg_original_accuracy:.1%}")
        print(f"   微调模型精确匹配: {custom_exact_matches}/{len(truth_results)} ({custom_exact_matches/len(truth_results):.1%})")
        print(f"   原始模型精确匹配: {original_exact_matches}/{len(truth_results)} ({original_exact_matches/len(truth_results):.1%})")
        
        if avg_custom_accuracy > avg_original_accuracy:
            improvement = avg_custom_accuracy - avg_original_accuracy
            print(f"\n🏆 微调模型整体更准确，准确率提升: {improvement:.1%}")
        elif avg_original_accuracy > avg_custom_accuracy:
            degradation = avg_original_accuracy - avg_custom_accuracy
            print(f"\n⚠️ 微调模型准确率下降: {degradation:.1%}")
        else:
            print(f"\n🤝 两个模型准确率相同")
        
        # 找出表现最好和最差的样本
        best_custom = max(truth_results, key=lambda x: x['custom_accuracy']['accuracy'])
        worst_custom = min(truth_results, key=lambda x: x['custom_accuracy']['accuracy'])
        
        print(f"\n📈 微调模型表现:")
        print(f"   最佳样本: {best_custom['audio_file']} (准确率: {best_custom['custom_accuracy']['accuracy']:.1%})")
        print(f"   最差样本: {worst_custom['audio_file']} (准确率: {worst_custom['custom_accuracy']['accuracy']:.1%})")
    
    # 找出最相似和最不同的结果
    best_similarity = max(results, key=lambda x: x['inter_model_metrics']['similarity'])
    worst_similarity = min(results, key=lambda x: x['inter_model_metrics']['similarity'])
    
    print(f"\n🔍 模型间一致性:")
    print(f"   最一致样本: {best_similarity['audio_file']} (相似度: {best_similarity['inter_model_metrics']['similarity']:.3f})")
    print(f"   最分歧样本: {worst_similarity['audio_file']} (相似度: {worst_similarity['inter_model_metrics']['similarity']:.3f})")
    
    # 调用深度分析
    analyze_finetuning_effectiveness(results)


def print_summary(results: List[Dict]) -> None:
    """
    向后兼容的总结函数
    """
    # 检查结果格式，决定使用哪个总结函数
    if results and any('inter_model_metrics' in r for r in results):
        print_summary_with_truth(results)
    else:
        # 原有的总结逻辑
        print(f"\n" + "="*80)
        print(f"📊 测试总结 (共 {len(results)} 个音频文件)")
        print(f"="*80)
        
        # 计算平均指标
        if results and 'metrics' in results[0]:
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


def diagnostic_test(model_path: str, audio_path: str, model_name: str) -> Dict:
    """
    诊断性测试，提供更详细的模型分析
    
    Args:
        model_path: 模型路径
        audio_path: 音频文件路径
        model_name: 模型名称
        
    Returns:
        dict: 详细的诊断结果
    """
    print(f"\n🔍 开始{model_name}诊断性测试...")
    
    # 加载模型
    model = load_single_model(model_path, model_name)
    
    # 获取模型基本信息
    print(f"📋 {model_name}基本信息:")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   设备: {model.device}")
    
    # 多次推理测试一致性
    print(f"\n🔄 进行3次推理测试一致性...")
    results = []
    times = []
    
    for i in range(3):
        start_time = time.time()
        result = model.transcribe(
            audio_path,
            language="zh",
            task="transcribe",
            temperature=0.0,  # 确定性推理
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            fp16=torch.cuda.is_available(),
        )
        inference_time = time.time() - start_time
        
        text = result['text'].strip()
        results.append(text)
        times.append(inference_time)
        
        print(f"   第{i+1}次: {text} ({inference_time:.2f}s)")
    
    # 检查一致性
    unique_results = set(results)
    is_consistent = len(unique_results) == 1
    avg_time = np.mean(times)
    
    print(f"✅ 一致性检查: {'通过' if is_consistent else '失败'}")
    print(f"📊 平均推理时间: {avg_time:.2f}s (±{np.std(times):.2f}s)")
    
    # 尝试不同参数的推理
    print(f"\n🧪 测试不同推理参数...")
    
    # 测试1：更高温度
    result_temp = model.transcribe(
        audio_path,
        language="zh", 
        task="transcribe",
        temperature=0.5,
        beam_size=1,
        condition_on_previous_text=False,
    )
    print(f"   高温度(0.5): {result_temp['text'].strip()}")
    
    # 测试2：beam search
    result_beam = model.transcribe(
        audio_path,
        language="zh",
        task="transcribe", 
        temperature=0.0,
        beam_size=5,
        condition_on_previous_text=False,
    )
    print(f"   Beam搜索(5): {result_beam['text'].strip()}")
    
    release_model(model, model_name)
    
    return {
        'model_name': model_name,
        'consistent': is_consistent,
        'avg_time': avg_time,
        'std_time': np.std(times),
        'results': results,
        'temp_result': result_temp['text'].strip(),
        'beam_result': result_beam['text'].strip(),
    }

def evaluate_finetuned_model_quality(model_path: str, test_audio_dir: str) -> Dict:
    """
    评估微调模型的整体质量
    
    Args:
        model_path: 微调模型路径
        test_audio_dir: 测试音频目录
        
    Returns:
        dict: 质量评估结果
    """
    print(f"\n🎯 开始微调模型质量评估...")
    print(f"📂 模型路径: {model_path}")
    print(f"📁 测试目录: {test_audio_dir}")
    
    if not os.path.exists(test_audio_dir):
        print(f"❌ 测试目录不存在: {test_audio_dir}")
        return {}
    
    # 查找测试音频文件
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(test_audio_dir).glob(f"*{ext}"))
    
    if not audio_files:
        print(f"❌ 在测试目录中未找到音频文件")
        return {}
    
    # 限制测试文件数量（避免测试时间过长）
    test_files = audio_files[:5]  # 只测试前5个文件
    print(f"📊 将测试 {len(test_files)} 个音频文件")
    
    # 加载模型
    model = load_single_model(model_path, "微调模型")
    
    results = []
    total_time = 0
    
    for i, audio_file in enumerate(test_files, 1):
        print(f"\n📝 测试文件 {i}/{len(test_files)}: {audio_file.name}")
        
        try:
            start_time = time.time()
            result = model.transcribe(
                str(audio_file),
                language="zh",
                task="transcribe",
                temperature=0.0,
                beam_size=1,
                condition_on_previous_text=False,
                fp16=torch.cuda.is_available(),
            )
            inference_time = time.time() - start_time
            total_time += inference_time
            
            text = result['text'].strip()
            segments = len(result['segments'])
            
            print(f"   结果: {text}")
            print(f"   推理时间: {inference_time:.2f}s")
            print(f"   段落数: {segments}")
            
            # 简单的质量检查
            quality_score = 0
            
            # 检查1：文本长度合理性
            if 5 <= len(text) <= 200:
                quality_score += 1
            
            # 检查2：包含中文字符
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                quality_score += 1
            
            # 检查3：不是重复字符
            if len(set(text)) > len(text) * 0.3:
                quality_score += 1
            
            # 检查4：推理时间合理
            if inference_time < 5.0:  # 小于5秒认为合理
                quality_score += 1
            
            results.append({
                'file': audio_file.name,
                'text': text,
                'time': inference_time,
                'segments': segments,
                'quality_score': quality_score
            })
            
        except Exception as e:
            print(f"❌ 测试 {audio_file.name} 时出错: {e}")
            continue
    
    release_model(model, "微调模型")
    
    if not results:
        print(f"❌ 没有成功的测试结果")
        return {}
    
    # 计算统计指标
    avg_time = total_time / len(results)
    avg_quality = np.mean([r['quality_score'] for r in results])
    avg_segments = np.mean([r['segments'] for r in results])
    avg_text_length = np.mean([len(r['text']) for r in results])
    
    print(f"\n📊 微调模型质量评估报告:")
    print(f"   测试文件数: {len(results)}")
    print(f"   平均推理时间: {avg_time:.2f}s")
    print(f"   平均质量得分: {avg_quality:.1f}/4.0")
    print(f"   平均段落数: {avg_segments:.1f}")
    print(f"   平均文本长度: {avg_text_length:.0f} 字符")
    
    # 质量判断
    if avg_quality >= 3.5:
        print(f"✅ 模型质量: 优秀")
    elif avg_quality >= 2.5:
        print(f"⚠️  模型质量: 良好")
    else:
        print(f"❌ 模型质量: 需要改进")
    
    return {
        'test_count': len(results),
        'avg_time': avg_time,
        'avg_quality': avg_quality,
        'avg_segments': avg_segments,
        'avg_text_length': avg_text_length,
        'results': results
    }

def load_ground_truth(audio_file: str, text_dir: str = None) -> Optional[str]:
    """
    加载音频文件对应的真实标签文本
    
    Args:
        audio_file: 音频文件路径
        text_dir: 文本文件目录，如果为None则自动推断
        
    Returns:
        str: 真实标签文本，如果找不到则返回None
    """
    # 获取音频文件名（不含扩展名）
    audio_path = Path(audio_file)
    audio_name = audio_path.stem
    
    # 如果没有指定文本目录，尝试自动推断
    if text_dir is None:
        # 尝试几种常见的文本目录结构
        possible_text_dirs = [
            audio_path.parent / "text",  # 同级text目录
            audio_path.parent / "../text",  # 上级text目录
            audio_path.parent.parent / "text",  # 上上级text目录
            Path("dataset/test/text"),  # 默认测试数据集路径
        ]
        
        for text_dir_candidate in possible_text_dirs:
            if text_dir_candidate.exists():
                text_dir = text_dir_candidate
                break
    
    if text_dir is None:
        return None
    
    # 查找对应的文本文件
    text_file = Path(text_dir) / f"{audio_name}.txt"
    
    if not text_file.exists():
        return None
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
        return ground_truth
    except Exception as e:
        print(f"⚠️ 读取真实标签文件 {text_file} 时出错: {e}")
        return None


def calculate_accuracy_metrics(pred_text: str, ground_truth: str) -> Dict[str, float]:
    """
    计算预测文本相对于真实标签的准确性指标
    
    Args:
        pred_text: 预测文本
        ground_truth: 真实标签文本
        
    Returns:
        dict: 准确性指标
    """
    if not pred_text.strip() or not ground_truth.strip():
        return {
            'wer': 1.0,
            'cer': 1.0,
            'accuracy': 0.0,
            'exact_match': False
        }
    
    try:
        # 计算错误率
        word_error_rate = wer(ground_truth, pred_text)
        char_error_rate = cer(ground_truth, pred_text)
        
        # 计算准确率（1 - 错误率）
        word_accuracy = max(0.0, 1.0 - word_error_rate)
        char_accuracy = max(0.0, 1.0 - char_error_rate)
        
        # 综合准确率
        overall_accuracy = (word_accuracy + char_accuracy) / 2
        
        # 精确匹配
        exact_match = pred_text.strip().lower() == ground_truth.strip().lower()
        
        return {
            'wer': word_error_rate,
            'cer': char_error_rate,
            'word_accuracy': word_accuracy,
            'char_accuracy': char_accuracy,
            'accuracy': overall_accuracy,
            'exact_match': exact_match
        }
    except Exception as e:
        print(f"⚠️ 计算准确性指标时出错: {e}")
        return {
            'wer': 1.0,
            'cer': 1.0,
            'word_accuracy': 0.0,
            'char_accuracy': 0.0,
            'accuracy': 0.0,
            'exact_match': False
        }

def main():
    """主函数"""
    # ===== 直接在代码中配置测试参数 =====
    # 模型路径配置
    custom_model_path = '../models/custom.pt'      # 微调模型路径
    original_model_path = '../models/large-v3.pt'  # 原始模型路径
    checkpoint_path = 'whisper-large-v3-finetuned/checkpoint-20000'  # HF checkpoint路径
    
    # 测试模式配置
    test_mode = "single"  # "single", "batch", "checkpoint_compare", "diagnostic", "quality_eval"
    
    if test_mode == "checkpoint_compare":
        # checkpoint vs custom.pt 对比模式
        audio_path = "../audio/00000034.wav"  # 修正：使用正确的音频文件路径
        audio_dir = None
    elif test_mode == "quality_eval":
        # 质量评估模式
        audio_path = None
        audio_dir = "dataset/test/audio"  # 使用测试数据集
    elif test_mode == "diagnostic":
        # 诊断模式
        audio_path = "../audio/00000034.wav"  # 修正：使用正确的音频文件路径
        audio_dir = None
    elif test_mode == "single":
        # 单文件测试模式
        audio_path = "../audio/00000034.wav"  # 修正：使用正确的音频文件路径
        audio_dir = None
    else:
        # 批量测试模式
        audio_path = None
        audio_dir = "dataset/test/audio"  # 使用有标签的测试数据集
    
    print("🚀 开始微调模型与原始模型对比测试")
    
    if test_mode == "checkpoint_compare":
        print(f"🔍 测试模式: Checkpoint vs Custom.pt 对比")
        print(f"⚙️  HF Checkpoint: {checkpoint_path}")
        print(f"⚙️  OpenAI Model: {custom_model_path}")
        print(f"🎵 音频文件: {audio_path}")
    elif test_mode == "quality_eval":
        print(f"🎯 测试模式: 微调模型质量评估")
        print(f"⚙️  微调模型路径: {custom_model_path}")
        print(f"📁 测试目录: {audio_dir}")
    elif test_mode == "diagnostic":
        print(f"🔍 测试模式: 诊断性测试")
        print(f"⚙️  微调模型路径: {custom_model_path}")
        print(f"⚙️  原始模型路径: {original_model_path}")
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
        elif test_mode == "quality_eval":
            # 微调模型质量评估
            quality_result = evaluate_finetuned_model_quality(custom_model_path, audio_dir)
            if quality_result:
                print(f"\n✅ 质量评估完成")
            else:
                print(f"\n❌ 质量评估失败")
        elif test_mode == "diagnostic":
            # 诊断性测试
            print(f"\n{'='*60} 微调模型诊断 {'='*60}")
            custom_diag = diagnostic_test(custom_model_path, audio_path, "微调模型")
            
            print(f"\n{'='*60} 原始模型诊断 {'='*60}")
            original_diag = diagnostic_test(original_model_path, audio_path, "原始模型")
            
            # 诊断总结
            print(f"\n{'='*60} 诊断总结 {'='*60}")
            print(f"🔍 微调模型一致性: {'✅ 通过' if custom_diag['consistent'] else '❌ 失败'}")
            print(f"🔍 原始模型一致性: {'✅ 通过' if original_diag['consistent'] else '❌ 失败'}")
            print(f"⏱️  平均推理时间对比:")
            print(f"   微调模型: {custom_diag['avg_time']:.2f}s (±{custom_diag['std_time']:.2f}s)")
            print(f"   原始模型: {original_diag['avg_time']:.2f}s (±{original_diag['std_time']:.2f}s)")
            
            if custom_diag['avg_time'] > original_diag['avg_time'] * 1.5:
                print(f"⚠️  微调模型推理速度异常缓慢，建议检查模型转换过程")
            
            # 检查不同参数下的结果一致性
            custom_variants = {custom_diag['results'][0], custom_diag['temp_result'], custom_diag['beam_result']}
            original_variants = {original_diag['results'][0], original_diag['temp_result'], original_diag['beam_result']}
            
            print(f"\n🧪 参数敏感性分析:")
            print(f"   微调模型输出变异数: {len(custom_variants)} 种")
            print(f"   原始模型输出变异数: {len(original_variants)} 种")
            
            if len(custom_variants) > 2:
                print(f"⚠️  微调模型输出不稳定，可能存在过拟合问题")
            
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

def analyze_finetuning_effectiveness(results: List[Dict]) -> None:
    """
    专门分析微调效果的函数
    
    Args:
        results: 测试结果列表
    """
    if not results:
        return
    
    print(f"\n" + "="*80)
    print(f"🎯 微调模型效果深度分析")
    print(f"="*80)
    
    # 统计有真实标签的结果
    truth_results = [r for r in results if r.get('ground_truth') and r.get('custom_accuracy') and r.get('original_accuracy')]
    
    if not truth_results:
        print(f"❌ 没有可用的真实标签数据进行分析")
        return
    
    # 分类统计
    custom_better = 0  # 微调模型更好
    original_better = 0  # 原始模型更好
    equal_performance = 0  # 性能相同
    
    custom_exact_wins = 0  # 微调模型精确匹配而原始模型错误
    original_exact_wins = 0  # 原始模型精确匹配而微调模型错误
    both_exact = 0  # 两者都精确匹配
    neither_exact = 0  # 两者都不精确匹配
    
    speed_comparisons = []
    accuracy_improvements = []
    
    for result in truth_results:
        custom_acc = result['custom_accuracy']['accuracy']
        original_acc = result['original_accuracy']['accuracy']
        
        custom_exact = result['custom_accuracy']['exact_match']
        original_exact = result['original_accuracy']['exact_match']
        
        # 统计准确率比较
        if custom_acc > original_acc:
            custom_better += 1
            accuracy_improvements.append(custom_acc - original_acc)
        elif original_acc > custom_acc:
            original_better += 1
            accuracy_improvements.append(original_acc - custom_acc)
        else:
            equal_performance += 1
        
        # 统计精确匹配情况
        if custom_exact and original_exact:
            both_exact += 1
        elif custom_exact and not original_exact:
            custom_exact_wins += 1
        elif original_exact and not custom_exact:
            original_exact_wins += 1
        else:
            neither_exact += 1
        
        # 统计速度比较
        speed_ratio = result['custom_time'] / result['original_time']
        speed_comparisons.append(speed_ratio)
    
    total_samples = len(truth_results)
    
    print(f"📊 准确率对比统计 ({total_samples} 个样本):")
    print(f"   🏆 微调模型更准确: {custom_better} 样本 ({custom_better/total_samples:.1%})")
    print(f"   🔄 原始模型更准确: {original_better} 样本 ({original_better/total_samples:.1%})")
    print(f"   🤝 准确率相同: {equal_performance} 样本 ({equal_performance/total_samples:.1%})")
    
    print(f"\n🎯 精确匹配统计:")
    print(f"   ✨ 微调模型精确匹配(原始错误): {custom_exact_wins} 样本 ({custom_exact_wins/total_samples:.1%})")
    print(f"   🔄 原始模型精确匹配(微调错误): {original_exact_wins} 样本 ({original_exact_wins/total_samples:.1%})")
    print(f"   🤝 两者都精确匹配: {both_exact} 样本 ({both_exact/total_samples:.1%})")
    print(f"   ❌ 两者都未精确匹配: {neither_exact} 样本 ({neither_exact/total_samples:.1%})")
    
    # 计算整体改善情况
    if custom_better > original_better:
        net_improvement = custom_better - original_better
        print(f"\n🎉 微调效果评估: 积极")
        print(f"   净改善样本数: {net_improvement} 个")
        print(f"   改善率: {net_improvement/total_samples:.1%}")
        
        if custom_exact_wins > 0:
            print(f"   特别亮点: {custom_exact_wins} 个样本从错误变为完全正确")
    
    elif original_better > custom_better:
        net_degradation = original_better - custom_better
        print(f"\n⚠️ 微调效果评估: 需要改进")
        print(f"   净退化样本数: {net_degradation} 个")
        print(f"   退化率: {net_degradation/total_samples:.1%}")
    else:
        print(f"\n🤝 微调效果评估: 中性")
        print(f"   整体准确率无显著变化")
    
    # 速度分析
    avg_speed_ratio = np.mean(speed_comparisons)
    speed_std = np.std(speed_comparisons)
    
    print(f"\n⚡ 推理速度分析:")
    print(f"   平均速度比例: {avg_speed_ratio:.2f}x (微调/原始)")
    print(f"   速度变化标准差: {speed_std:.2f}")
    
    if avg_speed_ratio > 1.5:
        print(f"   📝 速度评估: 明显变慢，建议优化")
        print(f"   💡 可能原因: 模型复杂度增加、转换精度损失")
    elif avg_speed_ratio > 1.2:
        print(f"   📝 速度评估: 轻微变慢，可接受范围")
    elif avg_speed_ratio > 0.8:
        print(f"   📝 速度评估: 变化不大，表现良好")
    else:
        print(f"   📝 速度评估: 反而变快，可能存在测量误差")
    
    # 综合评估建议
    print(f"\n💡 综合评估建议:")
    
    if custom_exact_wins > original_exact_wins and custom_better >= original_better:
        print(f"   ✅ 微调训练成功！模型在目标任务上表现更好")
        print(f"   📈 建议继续使用微调模型进行推理")
        if avg_speed_ratio > 1.3:
            print(f"   ⚡ 考虑进行模型量化或优化以改善推理速度")
    
    elif custom_better > original_better * 1.5:
        print(f"   ✅ 微调效果显著，准确率明显提升")
        print(f"   📊 准确率改善超过50%的样本占主导")
    
    elif custom_exact_wins == 0 and original_exact_wins > 0:
        print(f"   ⚠️ 微调可能存在问题，建议检查:")
        print(f"     • 训练数据质量和分布")
        print(f"     • 训练超参数设置")
        print(f"     • 模型转换过程")
    
    else:
        print(f"   📊 微调效果一般，建议进一步优化:")
        print(f"     • 增加训练数据量")
        print(f"     • 调整学习率和训练轮次")
        print(f"     • 检查数据标注质量")


if __name__ == "__main__":
    main()
