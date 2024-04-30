#!/usr/bin/env python
# coding: utf-8



import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments,DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import re




def preprocess_data(file_path):
    def remove_newlines(s):
        return re.sub(r'\n+', ' ', s)
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    train_texts = [item['train_text'] for item in data]
    train_texts = [remove_newlines(item) for item in train_texts]
    test_texts = [item['test_text'] for item in data]
    test_texts = [remove_newlines(item) for item in test_texts]

    return train_texts, test_texts

# Assuming your JSON data is stored in 'data.json'
file_path = 'LLM_review_dataset.json'
train_texts, test_texts = preprocess_data(file_path)

print(len(train_texts),len(test_texts))
train_texts_sampled=train_texts[:150000]
validation_texts_sampled=train_texts[-100:]
test_texts_sampled=test_texts[-100:]







model_name='gpt2-medium'
# Tokenize the texts
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

def encode_texts(text_list, tokenizer, max_length):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return inputs

# Define the maximum length
# max_length = 256  # or the maximum sequence length for your model
max_length = 1024

# Prepare the tokenized inputs for training
train_encodings = encode_texts(train_texts_sampled, tokenizer, max_length)
# validation_encodings = encode_texts(validation_texts_sampled, tokenizer, max_length)
test_encodings = encode_texts(test_texts_sampled, tokenizer, max_length)




class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Use the CustomDataset class
train_dataset = CustomDataset(train_encodings)
# validation_dataset = CustomDataset(validation_encodings)
test_dataset = CustomDataset(test_encodings)




model = GPT2LMHeadModel.from_pretrained(model_name)

# for name, module in model.named_modules():
#     print(name,module)




# Initialize the model
model = GPT2LMHeadModel.from_pretrained(model_name)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32, #Rank
    lora_alpha=32,
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
model.gradient_checkpointing_enable()

peft_model = get_peft_model(model, config)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"



# print(print_number_of_trainable_model_parameters(peft_model))



# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    overwrite_output_dir=False,
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=16,  # Batch size per device during training
    warmup_steps=10,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',           # Directory for storing logs
    save_total_limit=10,            # Keep only the last 10 model checkpoints
    save_steps=400,                  # Save model checkpoint every 500 steps
    evaluation_strategy="steps",     # Evaluate every `eval_steps`
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    optim="adamw_torch",
    fp16=True,
    tf32=True
)



# Then pass these to the Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train the model
trainer.train()




# Save the model
peft_model.save_pretrained('./model/train1024')



trainer.evaluate()

