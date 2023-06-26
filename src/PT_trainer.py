import json
import os
import re
import sqlite3
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer

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


conn = sqlite3.connect('data.db')

if not torch.cuda.is_available():
    print("Cuda is not available, training will be slow without cuda, switching to cpu")
else:
    print("Cuda is good")

torch.cuda.empty_cache()
print("Memory cache cleared")
conn.execute("DROP TABLE IF EXISTS data")
print("Database cleared")

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large').to(device)

conn.execute('''CREATE TABLE IF NOT EXISTS data
             (input_text TEXT NOT NULL,
             target_text TEXT NOT NULL);''')

num_files = len([filename for filename in os.listdir(dataset_dir) if not filename.startswith('validate-')])
print(f"Uploading {num_files} dataset files")
for filename in tqdm(os.listdir(dataset_dir), desc="Uploading dataset files", total=num_files):
    if not filename.startswith('validate-'):
        with open(f'{dataset_dir}/{filename}', 'r') as f:
            try:
                for item in json.load(f):
                    if 'input_text' in item and 'target_text' in item and item['target_text']:
                        conn.execute("INSERT INTO data (input_text, target_text) VALUES (?, ?)",
                                     (item['input_text'], item['target_text']))
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON in file {filename}")

batch_size = 1000

def data_generator():
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM data")
        num_rows = cursor.fetchone()[0]

        for offset in range(0, num_rows, batch_size):
            cursor.execute(f"SELECT * FROM data LIMIT {batch_size} OFFSET {offset}")
            rows = cursor.fetchall()
            for row in rows:
                yield row
    finally:
        cursor.close()

tokenizer = ByteLevelBPETokenizer()

train_encodings = tokenizer.encode_batch([item[0] for item in data_generator() if isinstance(item[0], str)])
train_labels = tokenizer.encode_batch([item[1] for item in data_generator() if isinstance(item[1], str)])



num_files = len([filename for filename in os.listdir(dataset_dir) if re.search(r'validate-', filename)])
print(f"Uploading {num_files} validation files")
for filename in tqdm(os.listdir(dataset_dir), desc="Uploading validation files", total=num_files):
    if re.search(r'validate-', filename):
        with open(f'{dataset_dir}/{filename}', 'r') as f:
            try:
                for item in json.load(f):
                    if 'input_text' in item and 'target_text' in item and item['target_text']:
                        conn.execute("INSERT INTO data (input_text, target_text) VALUES (?, ?)", (item['input_text'],
                                                                                                  item['target_text']))
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON in file {filename}")

eval_data = conn.execute("SELECT * FROM data").fetchall()

eval_encodings = tokenizer([item[0] for item in eval_data if isinstance(item[0], str)],
                           return_tensors='pt', padding=True,
                           truncation=True)
eval_labels = tokenizer([item[1] for item in eval_data if isinstance(item[1], str)],
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
    num_train_epochs=2,
)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
eval_dataloader = DataLoader(eval_dataset_combined, batch_size=1, shuffle=True, pin_memory=True)

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

conn.close()
