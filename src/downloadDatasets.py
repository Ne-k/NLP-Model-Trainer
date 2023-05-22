import json
import os

from datasets import load_dataset

gpt4 = load_dataset('vicgalle/alpaca-gpt4')

with open('../datasets/gpt4.json', 'a') as f:
    f.write('[\n')
    for example in gpt4['train']:
        input_text = example['instruction'] + " " + example['input']
        target_text = example['output']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

long_form_data = load_dataset('akoksal/LongForm')

with open('../datasets/LongForm.json', 'a') as f:
    f.write('[\n')
    for example in long_form_data['train']:
        input_text = example['input']
        target_text = example['output']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

long_form_data = load_dataset('akoksal/LongForm')

with open('../datasets/validate-LongForm.json', 'a') as f:
    f.write('[\n')
    for example in long_form_data['validation']:
        input_text = example['input']
        target_text = example['output']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

gsm8k_data = load_dataset('gsm8k', "main")

with open('../datasets/validate-gsm8k.json', 'a') as f:
    f.write('[\n')
    for example in gsm8k_data['test']:
        input_text = example['question']
        target_text = example['answer']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

codex_math_qa_data = load_dataset('theblackcat102/codex-math-qa')

with open('../datasets/validate-Codex.json', 'a') as f:
    f.write('[\n')
    for example in codex_math_qa_data['validation']:
        input_text = example['question']
        target_text = example['reply']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

codex_math_qa_rational_data = load_dataset('theblackcat102/codex-math-qa', "rational")

with open('../datasets/validate-CodexRational.json', 'a') as f:
    f.write('[\n')
    for example in codex_math_qa_rational_data['validation']:
        input_text = example['question']
        target_text = example['reply']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')


math = load_dataset('competition_math')

with open('../datasets/Competition_math.json', 'a') as f:
    f.write('[\n')
    for example in math['train']:
        input_text = example['problem']
        target_text = example['solution']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

math = load_dataset('competition_math')

with open('../datasets/validate-Competition_math.json', 'a') as f:
    f.write('[\n')
    for example in math['test']:
        input_text = example['problem']
        target_text = example['solution']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

piqa = load_dataset('piqa')

with open('../datasets/validate-Piqa.json', 'a') as f:
    f.write('[\n')
    for example in piqa['validation']:
        input_text = example['goal']
        target_text = example['sol1']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

human_eval_data = load_dataset('openai_humaneval')

with open('../datasets/OpenAI_HumanEval.json', 'a') as f:
    f.write('[\n')
    for example in human_eval_data['test']:
        input_text = example['prompt']
        target_text = example['canonical_solution']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

gsm8k_data = load_dataset('gsm8k', "main")

with open('../datasets/gsm8k.json', 'a') as f:
    f.write('[\n')
    for example in gsm8k_data['train']:
        input_text = example['question']
        target_text = example['answer']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

codex_math_qa_data = load_dataset('theblackcat102/codex-math-qa')

with open('../datasets/Codex.json', 'a') as f:
    f.write('[\n')
    for example in codex_math_qa_data['train']:
        input_text = example['question']
        target_text = example['reply']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

codex_math_qa_rational_data = load_dataset('theblackcat102/codex-math-qa', "rational")

with open('../datasets/CodexRational.json', 'a') as f:
    f.write('[\n')
    for example in codex_math_qa_rational_data['test']:
        input_text = example['question']
        target_text = example['reply']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

math = load_dataset('competition_math')

with open('../datasets/Competition_math.json', 'a') as f:
    f.write('[\n')
    for example in math['test']:
        input_text = example['problem']
        target_text = example['solution']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

spell = load_dataset('vishnun/SpellGram')

with open('../datasets/SpellGram.json', 'a') as f:
    f.write('[\n')
    for example in spell['train']:
        input_text = example['source']
        target_text = example['target']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

wikipedia_data = load_dataset('wikipedia', '20220301.en')

file_num1 = 1
counter1 = 0
f = open(f'../datasets/Wikipedia_{file_num1}.json', 'a')
f.write('[\n')

for example in wikipedia_data['train']:
    input_text = example['title']
    target_text = example['text']
    data = {'input_text': input_text, 'target_text': target_text}
    json.dump(data, f, indent=4)
    f.write(',\n')

    counter1 += 1
    if counter1 == 250:
        f.write(']\n')
        f.close()
        file_num1 += 1
        f = open(f'../datasets/Wikipedia{file_num1}.json', 'a')
        f.write('[\n')
        counter1 = 0

f.write(']\n')
f.close()

piqa = load_dataset('piqa')

with open('../datasets/Piqa.json', 'a') as f:
    f.write('[\n')
    for example in piqa['train']:
        input_text = example['goal']
        target_text = example['sol1']
        data = {'input_text': input_text, 'target_text': target_text}
        json.dump(data, f, indent=4)
        f.write(',\n')
    f.write(']\n')

wiki_qa = load_dataset('wiki_qa')

file_num = 1
counter = 0
f = open(f'../datasets/Wiki_qa_{file_num}.json', 'a')
f.write('[\n')

for example in wiki_qa['train']:
    input_text = example['question']
    target_text = example['answer']
    data = {'input_text': input_text, 'target_text': target_text}
    json.dump(data, f, indent=4)
    f.write(',\n')

    counter += 1
    if counter == 250:
        f.write(']\n')
        f.close()
        file_num += 1
        f = open(f'../datasets/Wiki_qa_{file_num}.json', 'a')
        f.write('[\n')
        counter = 0

f.write(']\n')
f.close()

print("Done saving datasets")
