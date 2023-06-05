import json
import os
import re
import tempfile

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, default_data_collator


class TG:
    def __init__(self, input_encoding, label_encoding):
        self.input_ids = input_encoding['input_ids']
        self.attention_mask = input_encoding['attention_mask']
        self.labels = label_encoding['input_ids']


class TextGenerationIterableDataset(IterableDataset):
    def __init__(self, ddir, tokenizer, max_length):
        self.dataset_dir = ddir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        num_items = 0
        for filename in os.listdir(self.dataset_dir):
            if not re.search(r'validate-', filename):
                with open(f'{self.dataset_dir}/{filename}', 'r') as f:
                    num_items += len(json.load(f))

        with tqdm(total=num_items, leave=True) as pbar:
            for filename in os.listdir(self.dataset_dir):
                if not re.search(r'validate-', filename):
                    with open(f'{self.dataset_dir}/{filename}', 'r') as f:
                        for item in json.load(f):
                            if 'input_text' in item and 'target_text' in item:
                                input_text = item['input_text']
                                label_text = item['target_text']
                                if input_text and label_text:
                                    input_encoding = self.tokenizer.encode_plus(
                                        input_text,
                                        max_length=self.max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt'
                                    )
                                    label_encoding = self.tokenizer.encode_plus(
                                        label_text,
                                        max_length=self.max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt'
                                    )
                                    pbar.update(1)
                                    yield (input_encoding, label_encoding)

    def __len__(self):
        num_items = 0
        for filename in os.listdir(self.dataset_dir):
            if not re.search(r'validate-', filename):
                with open(f'{self.dataset_dir}/{filename}', 'r') as f:
                    num_items += len(json.load(f))
        return num_items


if not torch.cuda.is_available():
    print("Cuda is not available, training will be slow without cuda, switching to cpu")
    print(torch.version.cuda)
else:
    print("Cuda is good")

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small').to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu"))

training_args = TrainingArguments(
    output_dir='../results',
    evaluation_strategy='epoch',
    logging_dir='../logs',
    per_device_train_batch_size=2,
    num_train_epochs=100,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
    fp16_opt_level='O2',
)

dataset_dir = './datasets'

print("Creating training and evaluation datasets...")
train_dataset = TextGenerationIterableDataset(dataset_dir, tokenizer, 512)
eval_dataset = TextGenerationIterableDataset(dataset_dir, tokenizer, 512)

print("Creating data loaders...")
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    num_workers=0
)

eval_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=2,
    num_workers=0
)

print("Creating trainer...")


def torch_default_data_collator(features):
    if hasattr(features[0], '__dict__'):
        features = [vars(f) for f in features]
    return default_data_collator(features)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=torch_default_data_collator
)

print("Starting training...")
trainer.train()

print("Saving trained model...")
model.save_pretrained('../trained_model')
