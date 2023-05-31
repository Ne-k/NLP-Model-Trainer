import json
from datasets import load_dataset
import os

os.makedirs('./datasets', exist_ok=True)
gpt4 = load_dataset('vicgalle/alpaca-gpt4')

with open('./datasets/gpt4.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(gpt4['train']):
        input_text = example['instruction'] + " " + example['input']
        target_text = example['output']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(gpt4['train']) - 1:
            f.write(',\n')
    f.write('\n]')

long_form_data = load_dataset('akoksal/LongForm')

with open('./datasets/LongForm.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(long_form_data['train']):
        input_text = example['input']
        target_text = example['output']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(long_form_data['train']) - 1:
            f.write(',\n')
    f.write('\n]')

long_form_data_val = load_dataset('akoksal/LongForm')

with open('./datasets/validate-LongForm.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(long_form_data_val['validation']):
        input_text = example['input']
        target_text = example['output']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(long_form_data_val['validation']) - 1:
            f.write(',\n')
    f.write('\n]')

gsm8k_data = load_dataset('gsm8k', "main")

with open('./datasets/validate-gsm8k.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(gsm8k_data['test']):
        input_text = example['question']
        target_text = example['answer']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(gsm8k_data['test']) - 1:
            f.write(',\n')
    f.write('\n]')

codex_math_qa_data_val = load_dataset('theblackcat102/codex-math-qa')

with open('./datasets/validate-Codex.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(codex_math_qa_data_val['validation']):
        input_text = example['question']
        target_text = example['reply']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(codex_math_qa_data_val['validation']) - 1:
            f.write(',\n')
    f.write('\n]')

codex_math_qa_rational_data_val = load_dataset('theblackcat102/codex-math-qa', "rational")

with open('./datasets/validate-CodexRational.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(codex_math_qa_rational_data_val['validation']):
        input_text = example['question']
        target_text = example['reply']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(codex_math_qa_rational_data_val['validation']) - 1:
            f.write(',\n')
    f.write('\n]')

# Currently broken, I'll maybe fix it
math = load_dataset('competition_math')

with open('./datasets/Competition_math.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(math['train']):
        input_text = example['problem']
        target_text = example['solution']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(math['train']) - 1:
            f.write(',\n')
    f.write('\n]')

math_val = load_dataset('competition_math')

with open('./datasets/validate-Competition_math.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(math_val['test']):
        input_text = example['problem']
        target_text = example['solution']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(math_val['test']) - 1:
            f.write(',\n')
    f.write(']\n')
piqa = load_dataset('piqa')

with open('./datasets/validate-Piqa.json', 'w') as f:
    f.write('[\n')
    for i, example in enumerate(piqa['validation']):
        input_text = example['goal']
        target_text = example['sol1']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(piqa['validation']) - 1:
            f.write(',\n')
    f.write('\n]\n')

human_eval_data = load_dataset('openai_humaneval')

with open('./datasets/OpenAI_HumanEval.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(human_eval_data['test']):
        input_text = example['prompt']
        target_text = example['canonical_solution']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(human_eval_data['test']) - 1:
            f.write(',\n')
    f.write('\n]')


gsm8k_data_val = load_dataset('gsm8k', 'main')

with open('./datasets/gsm8k.json', 'w') as f:
    f.write('[\n')
    for i, example in enumerate(gsm8k_data_val['train']):
        input_text = example['question']
        target_text = example['answer']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(gsm8k_data_val['train']) - 1:
            f.write(',\n')
    f.write('\n]')

codex_math_qa_data = load_dataset('theblackcat102/codex-math-qa')

with open('./datasets/Codex.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(codex_math_qa_data['train']):
        input_text = example['question']
        target_text = example['reply']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(codex_math_qa_data['train']) - 1:
            f.write(',\n')
    f.write(']\n')

codex_math_qa_rational_data = load_dataset('theblackcat102/codex-math-qa', 'rational')

with open('./datasets/CodexRational.json', 'w') as f:
    f.write('[\n')
    for i, example in enumerate(codex_math_qa_rational_data['test']):
        input_text = example['question']
        target_text = example['reply']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(codex_math_qa_rational_data['test']) - 1:
            f.write(',\n')
    f.write('\n]')


spell = load_dataset('vishnun/SpellGram')

with open('./datasets/SpellGram.json', 'a') as f:
    f.write('[\n')
    for i, example in enumerate(spell['train']):
        input_text = example['source']
        target_text = example['target']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(spell['train']) - 1:
            f.write(',\n')
    f.write('\n]')

piqa = load_dataset('piqa')

with open('./datasets/Piqa.json', 'w') as f:
    f.write('[\n')
    for i, example in enumerate(piqa['train']):
        input_text = example['goal']
        target_text = example['sol1']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(piqa['train']) - 1:
            f.write(',\n')
    f.write('\n]\n')

wiki_qa = load_dataset('wiki_qa')

with open('./datasets/Wiki_qa.json', 'w') as f:
    f.write('[\n')
    for i, example in enumerate(wiki_qa['train']):
        input_text = example['question']
        target_text = example['answer']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        if i != len(wiki_qa['train']) - 1:
            f.write(',\n')
    f.write('\n]\n')

wikipedia_data = load_dataset('wikipedia', '20220301.en')

file_num = 0
file_path = f'./datasets/Wikipedia_{file_num}.json'
file_count = 0

for i, example in enumerate(wikipedia_data['train']):
    # if file_count >= 500000: # I'm just throwing random numbers here because I don't want to calculate the math.
    #     # 20000 = 199 files.
    #     break
    if i % 100 == 0:
        file_num = i // 100
        file_path = f'./datasets/Wikipedia_{file_num}.json'
        with open(file_path, 'w') as f:
            f.write('[\n')
    input_text = example['title']
    target_text = example['text']
    data = {'input_text': input_text, 'target_text': target_text}
    with open(file_path, 'a') as f:
        json.dump(data, f, indent=4)
        if i != len(wikipedia_data['train']) - 1 and (i + 1) % 100 != 0:
            f.write(',\n')
        elif i != len(wikipedia_data['train']) - 1:
            f.write('\n]\n')
        # file_count += 1
            
# What the fuck? 
# if i % 100 != 99:
#     with open(file_path, 'a') as f:
#         f.write('\n]\n')

print("Done saving datasets")
