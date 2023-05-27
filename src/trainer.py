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
    print(torch.version.cuda)
else:
    print("Cuda is good")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

with tempfile.TemporaryDirectory() as temp_dir:
    temp_file_path = os.path.join(temp_dir, 'train_data.json')
    with open(temp_file_path, 'w+') as temp_file:
        for filename in os.listdir('./datasets'):
            if not re.search(r'validate-', filename):
                with open(f'./datasets/{filename}', 'r') as f:
                    try:
                        for item in json.load(f):
                            if 'input_text' in item and 'target_text' in item:
                                temp_file.write(json.dumps(item) + '\n')
                    except json.JSONDecodeError:
                        print(f"Error loading file: {filename}")
        temp_file.seek(0)
        train_data = [json.loads(line) for line in temp_file]

    train_encodings = tokenizer([item['input_text'] for item in train_data if isinstance(item['input_text'], str)],
                                return_tensors='pt', padding=True,
                                truncation=True)
    train_labels = tokenizer([item['target_text'] for item in train_data if isinstance(item['target_text'], str)],
                             return_tensors='pt', padding=True,
                             truncation=True)

    train_dataset = TextGenerationDataset(train_encodings, train_labels)

    temp_file_path = os.path.join(temp_dir, 'eval_data.json')
    with open(temp_file_path, 'w+') as temp_file:
        for filename in os.listdir('./datasets'):
            if re.search(r'validate-', filename):
                with open(f'./datasets/{filename}', 'r') as f:
                    try:
                        for item in json.load(f):
                            if 'input_text' in item and 'target_text' in item:
                                temp_file.write(json.dumps(item) + '\n')
                    except json.JSONDecodeError:
                        print(f"Error loading file: {filename}")
        temp_file.seek(0)
        eval_data = [json.loads(line) for line in temp_file]

    eval_encodings = tokenizer([item['input_text'] for item in eval_data if isinstance(item['input_text'], str)],
                               return_tensors='pt', padding=True,
                               truncation=True)
    eval_labels = tokenizer([item['target_text'] for item in eval_data if isinstance(item['target_text'], str)],
                            return_tensors='pt', padding=True,
                            truncation=True)

    eval_dataset = TextGenerationDataset(eval_encodings, eval_labels)

    eval_dataset_combined = ConcatDataset([train_dataset, eval_dataset])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset_combined, batch_size=8)

    training_args = TrainingArguments(
        output_dir='../results',
        evaluation_strategy='epoch',
        logging_dir='../logs',
        per_device_train_batch_size=1,
        num_train_epochs=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
        # save_strategy="no"
    )

    trainer.train()

    model.save_pretrained('./trained_model')