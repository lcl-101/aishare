### 使用 dataset-v2 当中的数据集进行训练
### dataset-v2 包括了两个文件夹，一个是 audio，一个是 text
### audio 文件夹中包含了 wav 格式的音频文件, text 文件夹中包含了对应的文本文件,它们的文件名是一一对应的
import torch
import os
import numpy as np
import json
import datetime
import random
import re
import unicodedata
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict
from datasets import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainingArguments
from huggingface_hub import login

#登录 huggingface hub
#login()
# 定义数据集路径
dataset_root_path = "dataset"
train_folder = "train"
test_folder = "test"
train_dataset_path = os.path.join(dataset_root_path, train_folder)
test_dataset_path = os.path.join(dataset_root_path, test_folder)

train_dataset_json_file = os.path.join(dataset_root_path, "train_dataset.json")
if os.path.exists(train_dataset_json_file):
    os.remove(train_dataset_json_file)
test_dataset_json_file = os.path.join(dataset_root_path, "test_dataset.json")
if os.path.exists(test_dataset_json_file):
    os.remove(test_dataset_json_file)

# 读取所有音频文件和文本文件,并且将文件名，audio 文件路径，text 文件路径保存到 dataset_json_file 文件中
train_audio_path = os.path.join(train_dataset_path, "audio")
train_text_path = os.path.join(train_dataset_path, "text")
for (dir_path, dir_name, file_name) in os.walk(train_audio_path):
    for name in file_name:
        audio_file_path = os.path.join(dir_path, name)
        text_file_path = os.path.join(train_text_path, name.replace('.wav', '.txt'))
        txt_sentence = open(text_file_path, encoding='UTF-8').read()
        name = name.replace('.wav', '')
        #将 name,audio_file_path，txt_sentence 保存到 dataset_json_file 文件中
        file = open(train_dataset_json_file, 'a', encoding='UTF-8')
        file.write(json.dumps({"name": name, "audio_file_path": audio_file_path, "txt_sentence": txt_sentence}, ensure_ascii=False) + '\n')
        file.close()

# 读取所有音频文件和文本文件,并且将文件名，audio 文件路径，text 文件路径保存到 dataset_json_file 文件中
test_audio_path = os.path.join(test_dataset_path, "audio")
test_text_path = os.path.join(test_dataset_path, "text")
for (dir_path, dir_name, file_name) in os.walk(test_audio_path):
    for name in file_name:
        audio_file_path = os.path.join(dir_path, name)
        text_file_path = os.path.join(test_text_path, name.replace('.wav', '.txt'))
        txt_sentence = open(text_file_path, encoding='UTF-8').read()
        name = name.replace('.wav', '')
        #将 name,audio_file_path，txt_sentence 保存到 dataset_json_file 文件中
        file = open(test_dataset_json_file, 'a', encoding='UTF-8')
        file.write(json.dumps({"name": name, "audio_file_path": audio_file_path, "txt_sentence": txt_sentence}, ensure_ascii=False) + '\n')
        file.close()


#使用 dataset 读取 dataset_json_file 文件，将数据集加载到内存中
#course_dataset = DatasetDict()
course_dataset = load_dataset('json', data_files={'train': train_dataset_json_file, 'test': test_dataset_json_file})

course_dataset = course_dataset.cast_column('audio_file_path', Audio(sampling_rate=16000))

#定义一个 whisper 分词器，用于将文本转换为模型可以接受的输入
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="zh", task="transcribe")
#定义一个特征提取器，用于将音频文件转换为模型可以接受的输入
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

#定义一个 prepare_dataset 函数，用于对数据集进行预处理
def prepare_dataset(batch):
    audio = batch['audio_file_path']
    batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
    batch['labels'] = tokenizer(batch['txt_sentence']).input_ids
    return batch

#使用 prepare_dataset 函数对数据集进行预处理
course_dataset = course_dataset.map(prepare_dataset, remove_columns=course_dataset.column_names["train"], num_proc=1)


# 完成了数据集的预处理之后，定义一个 whisper processor，准备进行训练
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="zh", task="transcribe")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

####### 定义 metric 计算函数
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# 定义一个训练函数
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.generation_config.language = "zh"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.suppress_tokens = []


training_args = Seq2SeqTrainingArguments(
    output_dir="example-model",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=10000,
    max_steps=50000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=10000,
    eval_steps=10000,
    logging_steps=25,
    report_to=None,
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)
processor.save_pretrained(training_args.output_dir)
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=course_dataset["train"],
    eval_dataset=course_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

#删除 json 文件
os.remove(train_dataset_json_file)
os.remove(test_dataset_json_file)

