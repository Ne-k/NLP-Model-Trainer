import json
import os
import re
import sqlite3
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch import nn, optim

dataset_dir = "./datasets"

class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1])
        return out

if not torch.cuda.is_available():
    print("Cuda is not available, training will be slow without cuda, switching to cpu")
else:
    print("Cuda is good")

torch.cuda.empty_cache()
print("Memory cache cleared")

tokenizer = ... # Create Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(input_size=tokenizer.vocab_size, hidden_size=256, output_size=1).to(device)

conn = sqlite3.connect('data.db')
conn.execute('''CREATE TABLE IF NOT EXISTS data
             (input_text TEXT NOT NULL,
             target_text TEXT NOT NULL);''')

num_files = len([filename for filename in os.listdir(dataset_dir) if not filename.startswith('validate-')])
print(f"Uploading {num_files} dataset files")
for filename in tqdm(os.listdir(dataset_dir), desc="Uploading dataset files", total=num_files):
    if not filename.startswith('validate-'):
        with open(f'{dataset_dir}/{filename}', 'r') as f:
            for item in json.load(f):
                if 'input_text' in item and 'target_text' in item and item['target_text']:
                    conn.execute("INSERT INTO data (input_text, target_text) VALUES (?, ?)", (item['input_text'], item['target_text']))

data = conn.execute("SELECT * FROM data").fetchall()
X = [item[0] for item in data if isinstance(item[0], str)]
y = [item[1] for item in data if isinstance(item[1], str)]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

train_encodings = tokenizer(X_train, return_tensors='pt', padding=True, truncation=True)
train_labels = torch.tensor(y_train, dtype=torch.float32)
train_dataset = LSTMDataset(train_encodings, train_labels)

val_encodings = tokenizer(X_val, return_tensors='pt', padding=True, truncation=True)
val_labels = torch.tensor(y_val, dtype=torch.float32)
val_dataset = LSTMDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

del data, X, y, X_train, X_val, y_train, y_val


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch['input_ids'].transpose(0, 1).to(device))
        loss = criterion(y_pred, y_batch.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch['input_ids'].transpose(0, 1).to(device))
            val_loss += criterion(y_pred, y_batch.to(device)).item()
        val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, train loss: {loss.item():.4f}, val loss: {val_loss:.4f}")


torch.save(model.state_dict(), 'my_model.pt')

conn.close()