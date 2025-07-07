"""
M6-2: Whisper 模型微调训练

使用自定义数据集对 Whisper 模型进行微调训练。

数据集结构:
- dataset/train/audio/ - 训练音频文件 (wav格式)  
- dataset/train/text/  - 训练文本文件 (txt格式)
- dataset/test/audio/  - 测试音频文件 (wav格式)
- dataset/test/text/   - 测试文本文件 (txt格式)

依赖安装:
pip install transformers datasets evaluate accelerate jiwer
"""
import torch
import os
import numpy as np
import json
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union

try:
    from transformers import (
        WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor,
        WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
    )
    from datasets import load_dataset, Audio
    from huggingface_hub import login
    import evaluate
except ImportError as e:
    print(f"❌ 缺少依赖包: {e}")
    print("请安装依赖: pip install torch transformers datasets evaluate huggingface_hub accelerate")
    exit(1)

# --- 配置参数 ---
DATASET_ROOT = "dataset"
TRAIN_FOLDER = "train"
TEST_FOLDER = "test"

# 模型路径（需提前下载到本地）
MODEL_NAME = "../models/whisper-large-v3"  # 本地模型路径，或改为 "openai/whisper-large-v3" 使用在线模型

LANGUAGE = "zh"
TASK = "transcribe"
OUTPUT_DIR = "whisper-large-v3-finetuned"

def create_dataset_json(folder_path, json_file_path):
    """创建数据集JSON文件"""
    print(f"📝 正在创建数据集文件: {json_file_path}")
    
    # 移除旧文件
    if os.path.exists(json_file_path):
        os.remove(json_file_path)
    
    audio_path = os.path.join(folder_path, "audio")
    text_path = os.path.join(folder_path, "text")
    
    if not os.path.exists(audio_path) or not os.path.exists(text_path):
        raise FileNotFoundError(f"数据集路径不存在: {folder_path}")
    
    count = 0
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        for audio_file in os.listdir(audio_path):
            if not audio_file.endswith('.wav'):
                continue
                
            audio_file_path = os.path.join(audio_path, audio_file)
            text_file_path = os.path.join(text_path, audio_file.replace('.wav', '.txt'))
            
            if not os.path.exists(text_file_path):
                print(f"⚠️ 缺少对应文本文件: {text_file_path}")
                continue
            
            try:
                with open(text_file_path, 'r', encoding='utf-8') as txt_file:
                    txt_sentence = txt_file.read().strip()
                
                name = audio_file.replace('.wav', '')
                data = {
                    "name": name,
                    "audio_file_path": audio_file_path,
                    "txt_sentence": txt_sentence
                }
                
                json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
                
            except Exception as e:
                print(f"❌ 处理文件 {audio_file} 时出错: {e}")
    
    print(f"✅ 完成，共处理 {count} 个样本")
    return count

def prepare_dataset(batch, feature_extractor, tokenizer):
    """数据预处理函数（支持批处理）"""
    if isinstance(batch['audio_file_path'], list):
        # 批处理模式
        input_features = []
        labels = []
        for audio in batch['audio_file_path']:
            features = feature_extractor(
                audio['array'], 
                sampling_rate=audio['sampling_rate']
            ).input_features[0]
            input_features.append(features)
        
        for text in batch['txt_sentence']:
            label = tokenizer(text).input_ids
            labels.append(label)
            
        batch['input_features'] = input_features
        batch['labels'] = labels
    else:
        # 单个样本模式
        audio = batch['audio_file_path']
        batch['input_features'] = feature_extractor(
            audio['array'], 
            sampling_rate=audio['sampling_rate']
        ).input_features[0]
        batch['labels'] = tokenizer(batch['txt_sentence']).input_ids
    
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """语音序列到序列数据整理器"""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 分离输入和标签，因为它们长度不同需要不同的填充方法
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 获取标记化的标签序列
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 用 -100 替换填充以正确忽略损失
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果在之前的标记化步骤中添加了 bos token，在这里删除它
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def compute_metrics(pred, tokenizer):
    """计算评估指标"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 用 pad_token_id 替换 -100
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # 解码预测和标签
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 计算 WER
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def monitor_gpu_utilization():
    """监控GPU利用率"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            utilization = int(result.stdout.strip())
            return utilization
    except:
        pass
    return None

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            utilization = monitor_gpu_utilization()
            
            print(f"🎮 GPU {i} ({gpu_name}):")
            print(f"   💾 总内存: {total_memory:.1f} GB")
            print(f"   🔥 已分配: {allocated:.1f} GB")
            print(f"   📦 已缓存: {cached:.1f} GB")
            print(f"   🆓 可用: {total_memory - cached:.1f} GB")
            if utilization is not None:
                print(f"   ⚡ GPU利用率: {utilization}%")
    else:
        print("❌ 未检测到CUDA设备")

def check_datasets_cache():
    """检查数据集缓存使用情况"""
    import shutil
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.exists(cache_dir):
        # 计算缓存大小
        total_size = shutil.disk_usage(cache_dir).used
        total_size_gb = total_size / (1024**3)
        
        print(f"📦 数据集缓存信息:")
        print(f"   📁 缓存目录: {cache_dir}")
        print(f"   💾 缓存大小: {total_size_gb:.1f} GB")
        
        # 检查具体的json缓存
        json_cache_dir = os.path.join(cache_dir, "json")
        if os.path.exists(json_cache_dir):
            json_dirs = [d for d in os.listdir(json_cache_dir) if d.startswith("default-")]
            print(f"   🗂️ JSON缓存目录数量: {len(json_dirs)}")
            
            if total_size_gb > 50:  # 超过50GB提醒
                print(f"   ⚠️ 缓存较大，可考虑清理旧缓存")
                print(f"   🧹 清理命令: rm -rf {cache_dir}/json/default-*")
        
        return total_size_gb
    else:
        print("📦 数据集缓存目录不存在")
        return 0

def clear_datasets_cache():
    """清理数据集缓存"""
    import shutil
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if not os.path.exists(cache_dir):
        print("📦 缓存目录不存在，无需清理")
        return 0
    
    # 计算清理前大小
    try:
        before_size = shutil.disk_usage(cache_dir).used / (1024**3)
    except:
        before_size = 0
    
    cleaned_count = 0
    cleaned_size = 0
    
    # 清理json缓存目录
    json_cache_dir = os.path.join(cache_dir, "json")
    if os.path.exists(json_cache_dir):
        for item in os.listdir(json_cache_dir):
            if item.startswith("default-"):
                item_path = os.path.join(json_cache_dir, item)
                try:
                    if os.path.isdir(item_path):
                        # 计算目录大小
                        for root, dirs, files in os.walk(item_path):
                            for file in files:
                                try:
                                    cleaned_size += os.path.getsize(os.path.join(root, file))
                                except:
                                    pass
                        shutil.rmtree(item_path)
                        cleaned_count += 1
                    else:
                        cleaned_size += os.path.getsize(item_path)
                        os.remove(item_path)
                        cleaned_count += 1
                except Exception as e:
                    print(f"   ⚠️ 清理 {item} 时出错: {e}")
    
    # 清理锁文件
    for item in os.listdir(cache_dir):
        if item.endswith(".lock") and "json_default-" in item:
            lock_path = os.path.join(cache_dir, item)
            try:
                os.remove(lock_path)
                cleaned_count += 1
            except Exception as e:
                print(f"   ⚠️ 清理锁文件 {item} 时出错: {e}")
    
    cleaned_size_gb = cleaned_size / (1024**3)
    
    if cleaned_count > 0:
        print(f"🧹 缓存清理完成:")
        print(f"   🗑️ 清理文件/目录数量: {cleaned_count}")
        print(f"   💾 释放空间: {cleaned_size_gb:.1f} GB")
    else:
        print("📦 没有找到需要清理的缓存文件")
    
    return cleaned_size_gb

def main():
    """主训练流程"""
    print("🚀 开始 Whisper 模型微调训练")
    print("=" * 60)
    
    # 检查GPU内存和缓存使用情况
    check_gpu_memory()
    print()
    
    # 自动清理数据集缓存
    print("🧹 自动清理数据集缓存...")
    cleared_size = clear_datasets_cache()
    print()
    
    # 检查清理后的缓存状态
    if cleared_size > 0:
        print("📦 清理后缓存状态:")
        check_datasets_cache()
        print()
    
    # 准备数据集路径
    train_dataset_path = os.path.join(DATASET_ROOT, TRAIN_FOLDER)
    test_dataset_path = os.path.join(DATASET_ROOT, TEST_FOLDER)
    train_json_file = os.path.join(DATASET_ROOT, "train_dataset.json")
    test_json_file = os.path.join(DATASET_ROOT, "test_dataset.json")
    
    try:
        # 创建数据集JSON文件
        print("📂 步骤 1: 准备数据集...")
        train_count = create_dataset_json(train_dataset_path, train_json_file)
        test_count = create_dataset_json(test_dataset_path, test_json_file)
        
        if train_count == 0 or test_count == 0:
            raise ValueError("数据集为空，请检查数据文件")
        
        # 加载数据集
        print("\n📥 步骤 2: 加载数据集...")
        dataset = load_dataset('json', data_files={
            'train': train_json_file, 
            'test': test_json_file
        })
        dataset = dataset.cast_column('audio_file_path', Audio(sampling_rate=16000))
        print(f"✅ 数据集加载完成 - 训练: {train_count}, 测试: {test_count}")
        
        # 初始化模型组件
        print("\n🔧 步骤 3: 初始化模型组件...")
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        print("✅ 模型组件初始化完成")
        
        # 数据预处理
        print("\n⚙️ 步骤 4: 数据预处理...")
        print("📝 缓存机制说明:")
        print("   🧹 自动清理: 每次启动时清理旧缓存，确保使用最新数据")
        print("   🔄 首次运行: 处理数据并缓存到 ~/.cache/huggingface/datasets/")
        print("   ⚡ 后续运行: 直接从缓存加载，速度极快")
        print("   🏷️ 缓存键值: 基于数据内容和处理函数的哈希值")
        print("   💾 存储格式: Apache Arrow 列式存储")
        print("   🚀 性能优化: 大批量预处理 + 内存固定传输")
        print("   📊 预期效果: 利用2TB内存优势，最大化单线程性能")
        
        def prepare_batch(batch):
            return prepare_dataset(batch, feature_extractor, tokenizer)
        
        dataset = dataset.map(
            prepare_batch, 
            remove_columns=dataset.column_names["train"], 
            num_proc=1,  # 单进程避免共享内存限制
            batched=True,  # 启用批处理
            batch_size=200,  # 增大批处理大小，利用大内存优势
            desc="🔄 处理音频特征和文本标签 (大批量优化)",
            load_from_cache_file=True,  # 启用缓存加速
        )
        print("✅ 数据预处理完成")
        
        # 设置数据整理器和评估函数
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        
        def compute_metrics_wrapped(pred):
            return compute_metrics(pred, tokenizer)
        
        # 加载模型
        print("\n🤖 步骤 5: 加载模型...")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.generation_config.language = LANGUAGE
        model.generation_config.task = TASK
        model.generation_config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        print("✅ 模型加载完成")
        
        # 训练参数
        print("\n⚙️ 步骤 6: 配置训练参数...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            
            # --- 批处理大小优化 (显存充足时可以增大) ---
            per_device_train_batch_size=20,        # 增大到20 (利用大内存和高显存)
            per_device_eval_batch_size=12,         # 增大到12
            gradient_accumulation_steps=3,         # 调整为3，有效批大小=20*3=60
            
            # --- 学习率和训练步数优化 ---
            learning_rate=5e-5,                    # 稍微提高学习率
            warmup_steps=2000,                     # 增加预热步数
            max_steps=20000,                       # 增加训练步数，更充分训练
            
            # --- 内存和计算优化 ---
            gradient_checkpointing=False,          # 禁用梯度检查点避免冲突
            fp16=False,                            # 关闭fp16
            bf16=True,                             # 使用bf16，精度更好
            dataloader_pin_memory=True,            # 启用pin_memory提升GPU传输速度
            dataloader_num_workers=0,              # 保持单进程避免共享内存不足
            # dataloader_prefetch_factor=2,        # 注释掉，因为workers=0时不需要
            
            # --- 评估和保存策略优化 ---
            eval_strategy="steps",
            eval_steps=2000,                       # 每2000步评估一次
            save_steps=2000,                       # 每2000步保存一次
            save_total_limit=3,                    # 只保留最近3个检查点
            load_best_model_at_end=True,           # 训练结束加载最佳模型
            
            # --- 生成参数优化 ---
            predict_with_generate=True,
            generation_max_length=448,             # 增加最大生成长度
            
            # --- 优化器和调度器 ---
            optim="adamw_torch",                   # 使用AdamW优化器
            weight_decay=0.01,                     # 添加权重衰减
            lr_scheduler_type="cosine",            # 使用余弦学习率调度
            
            # --- 日志和监控 ---
            logging_steps=100,                     # 增加日志频率
            report_to=None,                        # 可改为["tensorboard"]启用tensorboard
            
            # --- 其他优化设置 ---
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            remove_unused_columns=False,           # 保留所有列
            
            # --- 高显存专用设置 ---
            max_grad_norm=1.0,                     # 梯度裁剪
            warmup_ratio=0.1,                      # 预热比例
        )
        
        processor.save_pretrained(training_args.output_dir)
        print("✅ 训练参数配置完成")
        
        # 显示训练配置信息
        print(f"\n📊 训练配置信息:")
        print(f"   🎯 模型: {MODEL_NAME}")
        print(f"   💾 批大小: {training_args.per_device_train_batch_size} (训练) / {training_args.per_device_eval_batch_size} (评估)")
        print(f"   📈 有效批大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"   🔥 学习率: {training_args.learning_rate}")
        print(f"   🏃 训练步数: {training_args.max_steps}")
        print(f"   🎲 预热步数: {training_args.warmup_steps}")
        print(f"   💾 精度模式: {'BF16' if training_args.bf16 else 'FP16' if training_args.fp16 else 'FP32'}")
        print(f"   💿 输出目录: {training_args.output_dir}")
        
        # 估算训练时间 (基于实际每步时间)
        estimated_time_hours = (training_args.max_steps * 14.21) / 3600  # 14.21秒/步
        print(f"   ⏰ 预估训练时间: {estimated_time_hours:.1f} 小时 (基于14.21秒/步)")
        print()
        
        # 创建训练器
        print("\n🏋️ 步骤 7: 开始训练...")
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics_wrapped,
            processing_class=processor,  # 使用新的参数名
        )
        
        # 开始训练
        print("🚀 开始训练...")
        trainer.train()
        
        # 保存最终模型
        print("\n💾 步骤 8: 保存最终模型...")
        final_model_dir = f"{OUTPUT_DIR}/final"
        trainer.save_model(final_model_dir)
        processor.save_pretrained(final_model_dir)
        print(f"✅ 最终模型已保存至: {final_model_dir}")
        
        # 运行最终评估
        print("\n📊 步骤 9: 最终评估...")
        eval_results = trainer.evaluate()
        print("📋 最终评估结果:")
        for key, value in eval_results.items():
            print(f"   {key}: {value:.4f}")
        
        print("\n🎉 训练完成!")
        print(f"🎯 最佳模型WER: {eval_results.get('eval_wer', 'N/A'):.4f}")
        print(f"📁 模型保存位置: {final_model_dir}")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        raise
    
    finally:
        # 清理临时文件
        print("\n🧹 清理临时文件...")
        for file_path in [train_json_file, test_json_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ 已删除: {file_path}")

if __name__ == "__main__":
    main()

