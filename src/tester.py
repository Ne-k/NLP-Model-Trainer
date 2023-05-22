# Load the trained model
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('../backups/checkpoint-304500')

while True:
    # Get input text from the console
    input_text = input("Enter input text: ")

    # Check if the user wants to exit
    if input_text.lower() == 'exit':
        break

    # Generate a prediction for the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.9, top_k=50,
                                repetition_penalty=1.2)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True, )

    print(f"Output: {output_text}")
