import os
import torch
from datasets import Dataset, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import evaluate
from dataclasses import dataclass
from typing import Any

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "hi"
DATA_DIR = "data/calls"
OUTPUT_DIR = "./whisper-transport-finetuned"

def load_audio_dataset():
    audio_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.wav', '.mp3', '.m4a', '.ogg'))]
    data = {"audio": [], "sentence": []}
    for audio_file in audio_files:
        base = os.path.splitext(audio_file)[0]
        txt_file = os.path.join(DATA_DIR, base + ".txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            data["audio"].append(os.path.join(DATA_DIR, audio_file))
            data["sentence"].append(text)
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

dataset = load_audio_dataset()
dataset = dataset.train_test_split(test_size=0.1)

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

train_dataset = dataset["train"].map(prepare_dataset, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["test"].map(prepare_dataset, remove_columns=dataset["test"].column_names)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=400,
    save_steps=800,
    logging_steps=100,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Training completed!")