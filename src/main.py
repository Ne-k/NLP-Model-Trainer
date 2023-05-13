import json

import torch
from transformers \
    import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments


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


tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

with open('../TrainerData.json', 'r') as f:
    train_data = json.load(f)

train_encodings = tokenizer([item['input_text'] for item in train_data], return_tensors='pt', padding=True,
                            truncation=True)
train_labels = tokenizer([item['target_text'] for item in train_data], return_tensors='pt', padding=True,
                         truncation=True)
eval_dataset = TextGenerationDataset(train_encodings, train_labels)

train_dataset = TextGenerationDataset(train_encodings, train_labels)

training_args = TrainingArguments(
    output_dir='../results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    num_train_epochs=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
model.save_pretrained('trained_model')

with open('../TrainerData.json', 'r') as f:
    test_data = json.load(f)


def compute_accuracy(nlpModel, token, dataset):
    correct = 0

    for item in dataset:
        input_ids = item['input_ids'].unsqueeze(0)
        output_ids = nlpModel.generate(input_ids, max_new_tokens=100)
        output_text = token.decode(output_ids[0], skip_special_tokens=True)

        target_ids = item['labels']
        target_text = token.decode(target_ids, skip_special_tokens=True)

        if output_text == target_text:
            correct += 1

    acc = correct / len(dataset)

    return acc


test_encodings = tokenizer([item['input_text'] for item in test_data], return_tensors='pt', padding=True,
                           truncation=True)
test_labels = tokenizer([item['target_text'] for item in test_data], return_tensors='pt', padding=True, truncation=True)

test_dataset = TextGenerationDataset(test_encodings, test_labels)
accuracy = compute_accuracy(model, tokenizer, test_dataset)
metrics = trainer.evaluate(eval_dataset=test_dataset)

print('Metrics:', metrics)
print('Accuracy:', accuracy)
