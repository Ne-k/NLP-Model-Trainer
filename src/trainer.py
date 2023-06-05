import json
import os
import re
import tempfile

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments


class TG:
    def __init__(self, input_encoding, label_encoding):
        self.input_ids = input_encoding['input_ids']
        self.attention_mask = input_encoding['attention_mask']
        self.labels = label_encoding['input_ids']


class TextGenerationIterableDataset(IterableDataset):
    def __init__(self, ddir):
        self.dataset_dir = ddir

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
                                pbar.update(1)
                                yield item

    def __len__(self):
        return sum(1 for _ in self)


if not torch.cuda.is_available():
    print("Cuda is not available, training will be slow without cuda, switching to cpu")
    print(torch.version.cuda)
else:
    print("Cuda is good")

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small').to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # t5-base, google/flan-t5-base'

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

dataset_dir = '../datasets'

train_dataset = TextGenerationIterableDataset(dataset_dir)

eval_dataset = TextGenerationIterableDataset(dataset_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

model.save_pretrained('../trained_model')