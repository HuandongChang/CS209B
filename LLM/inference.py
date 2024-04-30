from HuggingFace_API import load_HF_model,load_HF_model_GPT2, generate_with_HF_model
import re
import numpy as np
import json
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer, 
    GPT2LMHeadModel
)

def generate_response(input, model, tokenizer):
    pred = generate_with_HF_model(tokenizer, model, input)
    pred = pred.split(input)[-1]
    return pred

# def preprocess_data(file_path, start_from=0):
#     def remove_newlines(s):
#         return re.sub(r'\n+', ' ', s)

#     with open(file_path, 'r') as file:
#         data = json.load(file)

#     train_texts = [item['train_text'] for item in data[start_from:] if len(item['train_text'].split(" "))<250]
#     test_texts = [item['test_text'] for item in data[start_from:] if len(item['test_text'].split(" "))<250]

#     train_review = [remove_newlines(item.split("Stars: ")[0]) for item in train_texts]
#     train_score = [item.split("Stars: ")[-1] for item in train_texts]
#     test_review = [remove_newlines(item.split("Stars: ")[0]) for item in test_texts]
#     True_score = [item.split("The stars of the comment is ")[-1] for item in train_texts]
#     # test_score = [item for item in test_texts]

#     return np.array(train_review), np.array(train_score), np.array(test_review), np.array(True_score)

def preprocess_data(file_path, start_from=0):
    def remove_newlines(s):
        return re.sub(r'\n+', ' ', s)

    with open(file_path, 'r') as file:
        data = json.load(file)

    train_texts = [item['train_text'] for item in data[start_from:] if len(item['train_text'].split(" "))<250]
    all_review = [remove_newlines(item.split("Stars: ")[0]) for item in train_texts]
    True_score = [item.split("The stars of the comment is ")[-1] for item in train_texts]
    # test_score = [item for item in test_texts]

    return np.array(all_review), np.array(True_score)


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


# model_ckpt = "google/gemma-2b-it"
# model_ckpt="openai-community/gpt2-medium"
# model_ckpt='./model/train1024'
GPT2="gpt2-medium"
tokenizer_GPT2, model_GPT2 = load_HF_model_GPT2(GPT2)

GPT2_FT='./model/train1024'
tokenizer_GPT2_FT, model_GPT2_FT = load_HF_model_GPT2(GPT2_FT)

Gemma = "google/gemma-2b-it"
tokenizer_Gemma, model_Gemma = load_HF_model(Gemma)


# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
# model = GPT2LMHeadModel.from_pretrained(model_ckpt)
model_GPT2.eval()
model_GPT2=model_GPT2.to('cuda')

model_GPT2_FT.eval()
model_GPT2_FT=model_GPT2_FT.to('cuda')

model_Gemma.eval()
model_Gemma=model_Gemma.to('cuda')


file_path = 'LLM_review_dataset.json'
all_review, true_score = preprocess_data(file_path, start_from=-2000)

fewshot_cot = read_txt("./fewshot_cot_short.txt")

# for i, review in tqdm(enumerate(test_review[:10]), total=len(test_review[:10])):
#     input_prompt = fewshot_cot + review + "\nStars:"
#     output = generate_response(input_prompt, model_GPT2, tokenizer_GPT2)
#     print("Example ", i)
#     print("input length is: ", len(input_prompt.split()))
#     print(input_prompt)
#     print("output length is: ", len(output.split()))
#     print(output)
#     print("\n\n\n")


# for i, review in tqdm(enumerate(test_review[:2]), total=len(test_review[:2])):
#     input_prompt = fewshot_cot + review + "\nStars:"
#     output_GPT2 = generate_response(input_prompt, model_GPT2, tokenizer_GPT2)
#     output_GPT2_FT = generate_response(input_prompt, model_GPT2_FT, tokenizer_GPT2_FT)
#     output_Gemma = generate_response(input_prompt, model_Gemma, tokenizer_Gemma)
#     print("Example ", i)
#     print("Input: \n", input_prompt)
#     print("\n\nGPT2 Output:\n", output_GPT2)
#     print("\n\nGPT2 Finetuned Output:\n", output_GPT2_FT)
#     print("\n\nGemma Output:\n", output_Gemma)
#     print("\n\n\n")



all_data = []  

for i, review in tqdm(enumerate(all_review), total=len(all_review)):
    input_prompt = fewshot_cot + review + "\nStars:"
    output_GPT2 = generate_response(input_prompt, model_GPT2, tokenizer_GPT2)
    output_GPT2_FT = generate_response(input_prompt, model_GPT2_FT, tokenizer_GPT2_FT)
    output_Gemma = generate_response(input_prompt, model_Gemma, tokenizer_Gemma)
    

    data = {
        "Example": i,
        "Review": review,
        "GPT2": output_GPT2,
        "GPT2_Finetuned": output_GPT2_FT,
        "Gemma": output_Gemma,
        "True_Score": true_score[i]
    }
   
    all_data.append(data)

with open('reviews_output.json', 'w') as f:
    json.dump(all_data, f, indent=4)








