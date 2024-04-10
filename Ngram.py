import re
import numpy as np
from pprint import pprint
import random
import time
import sys
from alive_progress import alive_bar
import copy
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name(0))


NGRAM_SIZE = 4
ngramDicts = {}

MODEL = 1


# Tokenize the input text and returns a list of tokens
def get_token_list(text: str) -> list:
    # lower case it
    text = text.lower()
    # tokenize hashtags
    text = re.sub(r"#(\w+)", r"<HASHTAG> ", text)
    text = re.sub(r'\d+(,(\d+))*(\.(\d+))?%?\s', '<NUMBER> ', text)
    # tokenize mentions
    text = re.sub(r"@(\w+)", r"<MENTION> ", text)
    # tokenize urls
    text = re.sub(r"http\S+", r"<URL> ", text)
    # starting with www
    text = re.sub(r"www\S+", r"<URL> ", text)

    special_chars = [' ', '*', '!', '?', '.', ',', ';', ':', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '-', '_', '—','=','+', '`', '~', '@', '#', '$', '%', '^', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8','9']
    # pad the special characters with spaces
    for char in special_chars:
        text = text.replace(char, ' ')
    # pad < and > with spaces
    text = text.replace('<', ' <')
    text = text.replace('>', '> ')

    return text.split()
# slipts the text into sentences and tokenizes them. 
def sentence_tokenizer(fullText: str, thresh: int) -> list:
    # lower case it
    fullText = fullText.lower()
  
    sentenceEnders = ['.', '!', '?']
    # split on sentence enders handling cases such as Mr. etc
    fullText = fullText.replace('mr.', 'mr')
    fullText = fullText.replace('mrs.', 'mrs')
    fullText = fullText.replace('dr.', 'dr')
    fullText = fullText.replace('st.', 'st')
    fullText = fullText.replace('co.', 'co')
    fullText = fullText.replace('inc.', 'inc')
    fullText = fullText.replace('e.g.', 'eg')
    fullText = fullText.replace('i.e.', 'ie')
    fullText = fullText.replace('etc.', 'etc')
    fullText = fullText.replace('vs.', 'vs')
    fullText = fullText.replace('u.s.', 'us')

    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', fullText)

    sentences = [s.replace('\n', ' ') for s in sentences]
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s != '']
    sentences = [get_token_list(s) for s in sentences]

    tokenDict = {}
    for sentence in sentences:
        for token in sentence:
            if token in tokenDict:
                tokenDict[token] += 1
            else:
                tokenDict[token] = 1

    for sentence in sentences:
        for i in range(len(sentence)):
            if tokenDict[sentence[i]] <= thresh:
                sentence[i] = '<unk>'

    return sentences

# replaces all tokens with frequency less than threshold with <unk>
def rem_low_freq(tokens: list, threshold: int) -> list:
    # get the frequency of each token
    freq = {}
    for token in tokens:
        if token in freq:
            freq[token] += 1
        else:
            freq[token] = 1

    # remove tokens with frequency less than threshold
    for token in list(freq.keys()):
        if freq[token] <= threshold:
            del freq[token]

    # replace all tokens not in freq with <unk>
    for i in range(len(tokens)):
        if tokens[i] not in freq:
            tokens[i] = '<unk>'

    return tokens

# constructs an n-gram dictionary from the input token list
def construct_ngram(n: int, token_list: list) -> dict:
    ngram_dict = {}
    for i in range(len(token_list) - n + 1):
        ngram_to_check = token_list[i:i + n]
        cur_dict = ngram_dict
        for j in range(n):
            if ngram_to_check[j] not in cur_dict:
                if j == n - 1:
                    cur_dict[ngram_to_check[j]] = 1
                else:
                    cur_dict[ngram_to_check[j]] = {}
            else:
                if j == n - 1:
                    cur_dict[ngram_to_check[j]] += 1
            cur_dict = cur_dict[ngram_to_check[j]]

    return ngram_dict
# counts the number of n-grams in the input n-gram dictionary using dfs
def dfs_count(ngram_dict: dict) -> int:
    count = 0
    for key, value in ngram_dict.items():
        if isinstance(value, dict):
            count += dfs_count(value)
        else:
            count += 1
    return count

# gives the count of the input n-gram
def ngram_count(ngram_dict: dict, ngram: list) -> int:
    cur_dict = ngram_dict[len(ngram)]
    if len(ngram) == 1:
        if ngram[0] in cur_dict:
            return cur_dict[ngram[0]]
        else:
            return cur_dict['<unk>']
    for i in range(len(ngram)):
        if ngram[i] in cur_dict:
            cur_dict = cur_dict[ngram[i]]
        else:
            return 0
    return cur_dict


dfs_countD = {}

def kneser_ney_smoothing(ngram_dict: dict, d: float, ngram: list) -> float:
    # Replace unknown tokens in ngram with '<unk>'
    ngram = ['<unk>' if token not in ngram_dict[1] else token.lower() for token in ngram]

    if len(ngram) == 1:
        # If unigram, compute probability directly
        denom = dfs_countD.get(2, dfs_count(ngram_dict[2]))
        count = sum(1 for value in ngram_dict[2].values() if ngram[-1] in value)
        return count/denom

    deno = ngram_count(ngram_dict, ngram[:-1])
    if(deno == 0):
        return 0
    else:
        first = max(ngram_count(ngram_dict, ngram) - d, 0) / deno
    
    try:
        cur_dict = ngram_dict[len(ngram)]
        for token in ngram[:-1]:
            cur_dict = cur_dict[token]
        second_rhs = len(cur_dict)
    except KeyError:
        second_rhs = 0
    
    second = d * second_rhs / ngram_count(ngram_dict, ngram[:-1])
    
    return first + second * kneser_ney_smoothing(ngram_dict, d, ngram[1:])



# returns witten bell smoothed probability of the input n-gram
def witten_bell_smoothing(ngram_dict: dict, ngram: list) -> float:
    # Replace unknown tokens in ngram with '<unk>'
    ngram = ['<unk>' if token not in ngram_dict[1] else token.lower() for token in ngram]

    if len(ngram) == 1:
        return ngram_count(ngram_dict, ngram) / len(ngram_dict[1])

    try:
        cur_dict = ngram_dict[len(ngram)]
        for token in ngram[:-1]:
            cur_dict = cur_dict[token]
        lambda_inv_num = len(cur_dict)
    except KeyError:
        lambda_inv_num = 0

    deno = ngram_count(ngram_dict, ngram[:-1]) + lambda_inv_num
    if(deno == 0):
        return 0
    else:
        lambda_val = lambda_inv_num / deno
        first = lambda_val * ngram_count(ngram_dict, ngram) / ngram_count(ngram_dict, ngram[:-1])
        second = (1 - lambda_val) * witten_bell_smoothing(ngram_dict, ngram[1:])
        return first + second
    # try:
    #     lambda_inv_num /= lambda_inv_num + ngram_count(ngram_dict, ngram[:-1])
    # except ZeroDivisionError:
    #     return 0

    # lambd = 1 - lambda_inv_num

    # first_term = lambd * ngram_count(ngram_dict, ngram) / ngram_count(ngram_dict, ngram[:-1])
    # second_term = lambda_inv_num * witten_bell_smoothing(ngram_dict, ngram[1:])

    # return first_term + second_term



# calculates the likelihood of the input sentence
def sentence_likelihood(ngram_dict: dict, sentence: list, smoothing: str, kneserd=0.75) -> float:
    tokens = sentence
    if smoothing == 'w' or smoothing == 'wb':
        likelihood = 0
        for i in range(len(tokens) - NGRAM_SIZE + 1):
            likelihood += np.log(max(witten_bell_smoothing(ngram_dict, tokens[i:i + NGRAM_SIZE]), 1e-15))
        return likelihood
    elif smoothing == 'k' or smoothing == 'kn':
        likelihood = 0
        for i in range(len(tokens) - NGRAM_SIZE + 1):
            likelihood += np.log(max(kneser_ney_smoothing(ngram_dict, kneserd, tokens[i:i + NGRAM_SIZE]), 1e-15))
        return likelihood

# calculates the perplexity of the input sentence
def perplexity(ngram_dict: dict, sentence: list, smoothing: str, kneserd=0.75) -> float:
    prob = sentence_likelihood(ngram_dict, sentence, smoothing, kneserd)
    prob = np.exp(prob)
    prob = max(prob, 1e-15)
    return pow(prob, -1 / len(sentence))


def get_plot(perp_list, file_name=None):
    plt.figure(figsize=(10, 6))
    plt.hist(perp_list, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Sentence Perplexities')
    plt.xlabel('Perplexity')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(file_name)
    plt.show()


def get_perp_plots(text, perp_text, train):
    sentences = copy.deepcopy(text)
    sentences = sentence_tokenizer(sentences, 1)
    sentences = [sentence for sentence in sentences if len(sentence) >= NGRAM_SIZE]
    Lines = sentences.copy()

    sentences2 = copy.deepcopy(perp_text)
    sentences2 = sentence_tokenizer(sentences2, 1)
    sentences2 = [sentence for sentence in sentences2 if len(sentence) >= NGRAM_SIZE]
    Lines2 = sentences2.copy()

    if(train == True):
        combined_text = ""
        for line in Lines:
            combined_text += " ".join(line) + " "

        tokens = rem_low_freq(get_token_list(combined_text), 1)

        for n in range(NGRAM_SIZE):
            ngramDicts[n + 1] = construct_ngram(n + 1, tokens)   

    wb_perplexities = []
    toWrite = []
    with alive_bar(len(Lines2)) as bar:
        for sentence in Lines2:
            wb_perplexities.append(perplexity(ngramDicts, sentence, 'wb'))
            toWrite.append(" ".join(sentence) + "\t" + str(wb_perplexities[-1]))
            bar()
    # Plotting
    get_plot(wb_perplexities, 'wb_plot_input.png')

    wb_avg = sum(wb_perplexities) / len(wb_perplexities)
    wb_median = np.median(wb_perplexities)
    print(f'Witten-Bell average perplexity: {wb_avg}')
    print(f'Witten-Bell median perplexity: {wb_median}')
    # write average perplexity at the top
    outputfile = open(f"LM{MODEL + 1}_train-perplexity.txt", "w", encoding="utf-8")
    outputfile.write(f'{wb_avg}\n')
    outputfile.write("\n".join(toWrite))
    outputfile.close()

    # print("strated")
    kn_perplexities = []
    toWrite = []
    with alive_bar(len(Lines2)) as bar:
        for sentence in Lines2:
            kn_perplexities.append(perplexity(ngramDicts, sentence, 'kn'))
            toWrite.append(" ".join(sentence) + "\t" + str(kn_perplexities[-1]))
            bar()

    kn_avg = sum(kn_perplexities) / len(kn_perplexities)
    kn_median = np.median(kn_perplexities)
    outputfile = open(f"LM{MODEL}_train-perplexity.txt", "w", encoding="utf-8")
    # write average perplexity at the top
    outputfile.write(f'{kn_avg}\n')
    outputfile.write("\n".join(toWrite))
    outputfile.close()
    print(f'Kneser-Ney average perplexity: {kn_avg}')
    print(f'Kneser-Ney median perplexity: {kn_median}')
    # plotting
    get_plot(kn_perplexities, 'kn_plot_input.png')

def get_sentences(csv_file):
    train_sentences = []
    test_sentences = []
    train_sentences2 = []
    
    chunksize = 10000
    first_20k_complete = False
    second_20k_counter = 0
    
    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        for _, row in chunk.iterrows():
            # Exclude null rows
            if pd.notnull(row['input']) and pd.notnull(row['output']):
                input_text, output_text = row['input'], row['output']
                
                # Check if still collecting first 20k 'output' sentences
                if not first_20k_complete:
                    if len(input_text) < 100 and len(test_sentences) < 20000:
                        test_sentences.append(input_text)
                    if len(output_text) < 100 and len(train_sentences) < 20000:
                        train_sentences.append(output_text)

                    if len(train_sentences) == 20000 and len(test_sentences) == 20000:
                        first_20k_complete = True
                # Start collecting second set of 20k 'output' sentences
                elif second_20k_counter < 20000 and len(output_text) < 100:
                    train_sentences2.append(output_text)
                    second_20k_counter += 1
            
            # Break if both sets of train sentences are collected
            if first_20k_complete and second_20k_counter == 20000:
                break
    
    return train_sentences, test_sentences, train_sentences2


sentences, test_sentences, output_sentences = get_sentences('C4_200M_1M.csv')


fullText = " ".join(sentences)
testText = " ".join(test_sentences)
perpText = " ".join(output_sentences)


print(len(sentences))
print(len(test_sentences))
print(len(output_sentences))


get_perp_plots(fullText, perpText, True)

get_perp_plots(testText, testText, False)
