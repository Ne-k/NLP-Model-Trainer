import json
import os
import tempfile
import torch
import re

from torch.utils.data import ConcatDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

torch.cuda.empty_cache()


class TextGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, encoding_file, label_file):
        self.encodings_file = encoding_file
        self.labels_file = label_file

    def __getitem__(self, idx):
        with open(self.encodings_file, 'r') as k:
            encodings = json.load(k)[idx]
        with open(self.labels_file, 'r') as k:
            labels = json.load(k)[idx]
        items = {key: val.tolist() for key, val in encodings.items()}
        items['labels'] = labels['input_ids'].tolist()
        return items

    def __len__(self):
        print(f"encodings_file: {self.encodings_file}")
        with tempfile.NamedTemporaryFile(mode='r', delete=False) as z:
            with open(self.encodings_file, 'r') as q:
                encodings = json.load(q)
                z.write(json.dumps(encodings))
                z.seek(0)
                encodings = json.load(z)
        return len(encodings)


if not torch.cuda.is_available():
    print("Cuda is not available, training will be slow without cuda, switching to cpu")
else:
    print("Cuda is good")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

with tempfile.TemporaryDirectory() as temp_dir:
    temp_file = os.path.join(temp_dir, 'train_data.json')
    for filename in os.listdir('./datasets'):
        if not re.search(r'validate-', filename):
            with open(f'./datasets/{filename}', 'r') as f:
                for item in json.load(f):
                    if 'input_text' in item and 'target_text' in item:
                        with open(temp_file, 'a') as temp_f:
                            temp_f.write(json.dumps(item) + '\n')
    train_data = []
    with open(temp_file, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as encodings_file:
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as labels_file:
            train_encodings = []
            train_labels = []
            for item in train_data:
                if isinstance(item['input_text'], str) and isinstance(item['target_text'], str):
                    encoding = tokenizer(item['input_text'], return_tensors='pt', padding=True, truncation=True)
                    label = tokenizer(item['target_text'], return_tensors='pt', padding=True, truncation=True)
                    train_encodings.append(encoding)
                    train_labels.append(label)
                    encodings_file.write(json.dumps(encoding.__dict__, indent=2, default=lambda x: x.tolist()) + '\n')
                    labels_file.write(json.dumps(label.__dict__, default=lambda x: x.tolist()) + '\n')
            encodings_file.seek(0)
            labels_file.seek(0)
            train_dataset = TextGenerationDataset(encodings_file.name, labels_file.name)

    temp_file = os.path.join(temp_dir, 'eval_data.json')
    for filename in os.listdir('./datasets'):
        if re.search(r'validate-', filename):
            with open(f'./datasets/{filename}', 'r') as f:
                for item in json.load(f):
                    if 'input_text' in item and 'target_text' in item:
                        with open(temp_file, 'a') as temp_f:
                            temp_f.write(json.dumps(item) + '\n')
    eval_data = []
    with open(temp_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as encodings_file:
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as labels_file:
            eval_encodings = []
            eval_labels = []
            for item in eval_data:
                if isinstance(item['input_text'], str) and isinstance(item['target_text'], str):
                    encoding = tokenizer(item['input_text'], return_tensors='pt', padding=True, truncation=True)
                    label = tokenizer(item['target_text'], return_tensors='pt', padding=True, truncation=True)
                    eval_encodings.append(encoding.__dict__)
                    eval_labels.append(label.__dict__)
                    encodings_file.write(json.dumps(encoding.__dict__, indent=2, default=lambda x: x.tolist()) + '\n')
                    labels_file.write(json.dumps(label.__dict__, default=lambda x: x.tolist()) + '\n')
            encodings_file.seek(0)
            labels_file.seek(0)
            eval_dataset = TextGenerationDataset(encodings_file.name, labels_file.name)

    eval_dataset_combined = ConcatDataset([train_dataset, eval_dataset])

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        per_device_train_batch_size=1,
        num_train_epochs=100,
        gradient_accumulation_steps=2,
        fp16=True
    )


    def checkpoint_fn(*inputs):
        return model(*inputs)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_combined,
        save_strategy='no',
        checkpoint_fn=checkpoint_fn
    )

    trainer.train()

    model.save_pretrained('../trained_model')
