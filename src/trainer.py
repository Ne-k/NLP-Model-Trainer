import json
import os
import tempfile
import torch
import re

from torch.utils.data import ConcatDataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments


class TextGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return len(self.labels['input_ids'])


if not torch.cuda.is_available():
    print("Cuda is not available, training will be slow without cuda, switching to cpu")
else:
    print("Cuda is good")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

with tempfile.TemporaryDirectory() as temp_dir:
    with tempfile.NamedTemporaryFile(mode='w+', dir=temp_dir, delete=False) as train_encodings_file, \
            tempfile.NamedTemporaryFile(mode='w+', dir=temp_dir, delete=False) as train_labels_file, \
            tempfile.NamedTemporaryFile(mode='w+', dir=temp_dir, delete=False) as eval_encodings_file, \
            tempfile.NamedTemporaryFile(mode='w+', dir=temp_dir, delete=False) as eval_labels_file:

        for filename in os.listdir('./datasets'):
            if not re.search(r'validate-', filename):
                with open(f'./datasets/{filename}', 'r') as f:
                    try:
                        for item in json.load(f):
                            if 'input_text' in item and 'target_text' in item:
                                train_encodings_file.write(json.dumps(item['input_text']) + '\n')
                                train_labels_file.write(json.dumps(item['target_text']) + '\n')
                    except json.JSONDecodeError:
                        print(f"Error loading file: {filename}")

            elif re.search(r'validate-', filename):
                with open(f'./datasets/{filename}', 'r') as f:
                    try:
                        for item in json.load(f):
                            if 'input_text' in item and 'target_text' in item:
                                eval_encodings_file.write(json.dumps(item['input_text']) + '\n')
                                eval_labels_file.write(json.dumps(item['target_text']) + '\n')
                    except json.JSONDecodeError:
                        print(f"Error loading file: {filename}")

        train_encodings_file.seek(0)
        train_labels_file.seek(0)
        eval_encodings_file.seek(0)
        eval_labels_file.seek(0)

        train_encodings = tokenizer(train_encodings_file.readlines(), return_tensors='pt', padding=True, truncation=True)
        train_labels = tokenizer(train_labels_file.readlines(), return_tensors='pt', padding=True, truncation=True)
        eval_encodings = tokenizer(eval_encodings_file.readlines(), return_tensors='pt', padding=True, truncation=True)
        eval_labels = tokenizer(eval_labels_file.readlines(), return_tensors='pt', padding=True, truncation=True)

        train_dataset = TextGenerationDataset(train_encodings, train_labels)
        eval_dataset = TextGenerationDataset(eval_encodings, eval_labels)

        eval_dataset_combined = ConcatDataset([train_dataset, eval_dataset])

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset_combined, batch_size=8, shuffle=True)

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            logging_dir='./logs',
            per_device_train_batch_size=1,
            num_train_epochs=100,
            save_strategy='epoch'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset
        )

        trainer.train()

        model.save_pretrained('./trained_model')