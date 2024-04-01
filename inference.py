from HuggingFace_API import load_HF_model, generate_with_HF_model
import re
import numpy as np
import json
from tqdm import tqdm

def generate_response(input, model, tokenizer):
    pred = generate_with_HF_model(tokenizer, model, input)
    pred = pred.split(input)[-1]
    return pred

def preprocess_data(file_path):
    def remove_newlines(s):
        return re.sub(r'\n+', ' ', s)

    with open(file_path, 'r') as file:
        data = json.load(file)

    train_texts = [item['train_text'] for item in data]
    test_texts = [item['test_text'] for item in data]
    train_review = [remove_newlines(item.split("Stars: ")[0]) for item in train_texts]
    train_score = [item.split("Stars: ")[-1] for item in train_texts]
    test_review = [remove_newlines(item.split("Stars: ")[0]) for item in test_texts]
    test_score = [item.split("Stars: ")[-1] for item in test_texts]

    return np.array(train_review), np.array(train_score), np.array(test_review), np.array(test_score)

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


model_ckpt = "google/gemma-2b-it"
tokenizer, model = load_HF_model(model_ckpt)

file_path = 'LLM_review_dataset.json'
_, _, test_review, _ = preprocess_data(file_path)

fewshot_cot = read_txt("./prompt.txt")

for i, review in tqdm(enumerate(test_review), total=len(test_review)):
    input_prompt = fewshot_cot + review + "\nStars:"
    output = generate_response(input_prompt, model, tokenizer)
    print(output)
    print("")













