import csv
import os
import random
from itertools import chain
from string import punctuation

import datasets
import nltk
import numpy as np
import pandas as pd
import torch
import time
from datasets import Dataset as dDataset
from datasets import load_metric
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pickle as pkl
from tqdm import tqdm
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer, BertTokenizer, BertModel)
from jiwer import wer
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = 't5-base'

pd.set_option('display.max_colwidth', None)

if not os.path.exists("dataframePandasGrammar.csv"):
    # Load data from the CSV file
    df = pd.read_csv('C4_200M_1M.csv', header=None)

    # Limit the data to the first 1,000,000 rows
    df = df.iloc[:100000]

    # Rename columns if necessary
    df.columns = ["input", "output"]

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv("dataframePandasGrammar.csv", index=False)
else:
    # Load the DataFrame from the CSV file
    df = pd.read_csv("dataframePandasGrammar.csv")



tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)


train_df, val_df = train_test_split(df, test_size=0.20, shuffle=True)
train_df, test_df = train_test_split(train_df, test_size=0.20, shuffle=True)


def calc_token_len(example):
    return len(tokenizer(example).input_ids)
val_df['input_token_len'] = val_df['input'].apply(calc_token_len)


train_dataset = dDataset.from_pandas(train_df)
test_dataset = dDataset.from_pandas(val_df)


class LangDataset(Dataset):
    def __init__(self, dataset, tokenizer, print_text=False):
        self.dataset = dataset
        self.maxPad = False
        self.tokenizer = tokenizer
        self.max_len = 64

    def __len__(self):
        return len(self.dataset)

    def tokenize_data(self, example):
        input_, target_ = example['input'], example['output']

        # tokenize inputs
        tokenized_inputs = tokenizer(input_, pad_to_max_length=self.maxPad,
                                     max_length=self.max_len,
                                     return_attention_mask=True)

        tokenized_targets = tokenizer(target_, pad_to_max_length=self.maxPad,
                                      max_length=self.max_len,
                                      return_attention_mask=True)

        inputs = {"input_ids": tokenized_inputs['input_ids'],
                  "attention_mask": tokenized_inputs['attention_mask'],
                  "labels": tokenized_targets['input_ids']
                  }

        return inputs

    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])

        return inputs

dataset = LangDataset(test_dataset, tokenizer, True)



data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding='longest', return_tensors='pt')


# defining training related arguments
batch_size = 32
args = Seq2SeqTrainingArguments(output_dir="./weights",
                                evaluation_strategy="steps",
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                learning_rate=2e-5,
                                num_train_epochs=2,
                                weight_decay=0.01,
                                save_total_limit=10,
                                predict_with_generate=True,
                                fp16=True,
                                gradient_accumulation_steps=5,
                                eval_steps=9000,
                                save_steps=2000
                                )
                
trainer = Seq2SeqTrainer(model=model,
                         args=args,
                         train_dataset=LangDataset(
                             train_dataset, tokenizer),
                         eval_dataset=LangDataset(test_dataset, tokenizer),
                         tokenizer=tokenizer,
                         data_collator=data_collator)

if not os.path.exists('t5'):
    trainer.train()
    trainer.save_model('t5')

print('Loading model')
model = T5ForConditionalGeneration.from_pretrained('./t5/')
model.to(device)


def correct_grammar(input_text, num_return_sequences):
    batch = tokenizer([input_text], truncation=True, padding='max_length',
                      max_length=64, return_tensors="pt")
    batch = batch.to(device)
    translated = model.generate(**batch, max_length=64, num_beams=4,
                                num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

# ## Testing

if not os.path.exists('./grammarOutput.csv'):
    test_df = test_df[:1000]
    grammarOutputFile = open('./grammarOutput.csv', 'w')
    grammarOutputWriter = csv.DictWriter(
        grammarOutputFile, fieldnames=['input', 'output', 'truth'])
    grammarOutputWriter.writeheader()

    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        inp = row['input']
        prediction = correct_grammar(inp, num_return_sequences=1)[0]
        grammarOutputWriter.writerow(
            {'input': inp, 'output': prediction, 'truth': row['output']})

    grammarOutputFile.close()
