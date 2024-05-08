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
from icecream import ic
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer, BertTokenizer, BertModel)
from jiwer import wer
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import csv
from tqdm import tqdm
from bert_score import score as bert_score
from jiwer import wer
from rouge_score import rouge_scorer


scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
total_rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
total_rouge_scores_input = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
total_bert_scores = {"precision": 0, "recall": 0, "f1": 0}
total_bert_scores_input = {"precision": 0, "recall": 0, "f1": 0}

grammarFile = open('./grammarOutput.csv', 'r')
grammarReader = csv.DictReader(grammarFile)
total_wer = 0
total_input_wer = 0
correct = []
predicted = []
Inps = []
data = list(grammarReader)


for row in tqdm(data, desc="Calculating Stats"):
    truth = row['truth']
    prediction = row['output']
    inp = row['input']

    correct.append(truth)
    predicted.append(prediction)
    Inps.append(inp)

    # Calculating ROUGE scores
    rougeScore = scorer.score(truth, prediction)
    inputRougeScore = scorer.score(truth, inp)

    for key in rougeScore:
        total_rouge_scores[key] += rougeScore[key].fmeasure
        total_rouge_scores_input[key] += inputRougeScore[key].fmeasure

    # Calculating WER
    total_wer += wer(truth, prediction)
    total_input_wer += wer(truth, inp)

    # Calculating BERTScore
   
    precision, recall, f1 = bert_score([prediction], [truth], lang='en')
    total_bert_scores['precision'] += precision.mean().item()
    total_bert_scores['recall'] += recall.mean().item()
    total_bert_scores['f1'] += f1.mean().item()

    precision_input, recall_input, f1_input = bert_score([inp], [truth], lang='en')
    total_bert_scores_input['precision'] += precision_input.mean().item()
    total_bert_scores_input['recall'] += recall_input.mean().item()
    total_bert_scores_input['f1'] += f1_input.mean().item()

# Printing WER
print('WER: ', total_wer/len(data))
print('input WER: ', total_input_wer/len(data))
percentChange = (total_wer-total_input_wer)/total_input_wer
print(f"Percent change: {percentChange*100}%")

# Printing ROUGE scores
avgRouge = {key: total_rouge_scores[key]/len(data) for key in total_rouge_scores}
avgRougeinput = {key: total_rouge_scores_input[key]/len(data) for key in total_rouge_scores_input}
print('Avg ROUGE: ', avgRouge)
print('Avg input ROUGE: ', avgRougeinput)
percChangeRouge = {key: 100 * (avgRouge[key]-avgRougeinput[key])/avgRougeinput[key] for key in avgRouge}
print(f'Percent change ROUGE: {percChangeRouge}')

# Printing BERTScore
avg_bert_scores = {key: total_bert_scores[key] / len(data) for key in total_bert_scores}
avg_bert_scores_input = {key: total_bert_scores_input[key] / len(data) for key in total_bert_scores_input}
print('Avg BERTScore: ', avg_bert_scores)
print('Avg BERTScore (input): ', avg_bert_scores_input)



bleuCorrect = [[text.split()] for text in correct]
bleuPredicted = [text.split() for text in predicted]
blueInps = [text.split() for text in Inps]
actualBleu = corpus_bleu(bleuCorrect, bleuPredicted)
print('BLEU: ', actualBleu)

inputBleu = corpus_bleu(bleuCorrect, blueInps)
print("input BLEU: ", inputBleu)

print(f"Percent change: {((actualBleu-inputBleu)/inputBleu)*100}%")
