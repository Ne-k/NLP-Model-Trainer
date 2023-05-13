import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('./trained_model')

while True:
    user_input = input('Enter text to generate: ')
    if user_input.lower() == 'quit':
        break

    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output_ids = model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.9, top_k=50,
                                repetition_penalty=1.2)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(output_text)
