import json
import os
import tempfile
import torch
import re
from tqdm import tqdm, trange

from torch.utils.data import ConcatDataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

dataset_dir = "./datasets"


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

torch.cuda.empty_cache()
print("Memory cache cleared")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
    num_files = len([filename for filename in os.listdir(dataset_dir) if not filename.startswith('validate-')])
    print(f"Uploading {num_files} dataset files")
    for filename in tqdm(os.listdir(dataset_dir), desc="Uploading dataset files", total=num_files):
        if not filename.startswith('validate-'):
            with open(f'{dataset_dir}/{filename}', 'r') as f:
                for item in json.load(f):
                    if 'input_text' in item and 'target_text' in item:
                        temp_file.write(json.dumps(item) + '\n')
    temp_file.seek(0)
    train_data = [json.loads(line) for line in temp_file]

train_encodings = tokenizer([item['input_text'] for item in train_data if isinstance(item['input_text'], str)],
                            return_tensors='pt', padding=True,
                            truncation=True)
train_labels = tokenizer([item['target_text'] for item in train_data if isinstance(item['target_text'], str)],
                         return_tensors='pt', padding=True,
                         truncation=True)

train_dataset = TextGenerationDataset(train_encodings, train_labels)

del train_data

with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
    num_files = len([filename for filename in os.listdir(dataset_dir) if re.search(r'validate-', filename)])
    print(f"Uploading {num_files} validation files")
    for filename in tqdm(os.listdir(dataset_dir), desc="Uploading validation files", total=num_files):
        if re.search(r'validate-', filename):
            with open(f'{dataset_dir}/{filename}', 'r') as f:
                for item in json.load(f):
                    if 'input_text' in item and 'target_text' in item:
                        temp_file.write(json.dumps(item) + '\n')
    temp_file.seek(0)
    eval_data = [json.loads(line) for line in temp_file]

eval_encodings = tokenizer([item['input_text'] for item in eval_data if isinstance(item['input_text'], str)],
                           return_tensors='pt', padding=True,
                           truncation=True)
eval_labels = tokenizer([item['target_text'] for item in eval_data if isinstance(item['target_text'], str)],
                        return_tensors='pt', padding=True,
                        truncation=True)

eval_dataset = TextGenerationDataset(eval_encodings, eval_labels)

del eval_data

eval_dataset_combined = ConcatDataset([train_dataset, eval_dataset])

training_args = TrainingArguments(
    output_dir='../results',
    evaluation_strategy='epoch',
    logging_dir='../logs',
    per_device_train_batch_size=1,
    num_train_epochs=100,
)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_dataloader = DataLoader(eval_dataset_combined, batch_size=1, shuffle=True)

print("\033[93mStarting training...\033[0m")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader.dataset,
    eval_dataset=eval_dataloader.dataset
)

with torch.no_grad():
    trainer.evaluate()

trainer.train()

model.save_pretrained('../trained_model')
