# import all necessary packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import Counter
import random
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import collections
import itertools
import re
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from nltk.stem import WordNetLemmatizer
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from numpy import dot
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, matthews_corrcoef
)


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# print(torch.cuda.get_device_name(0))


# ## Data Preprocessing


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


# print(len(sentences))
# print(len(test_sentences))
# print(len(output_sentences))
# print(sentences[:10])


class Preprocess():
    def __init__(self, sentences):
        self.sentences = sentences
        self.stop_words = set(stopwords.words('english'))

    def word_tokenizer(self):
        # Define unique placeholders for special tokens
        placeholders = {
            '<URL>': 'URL_PLACEHOLDER',
            '<MAILID>': 'MAILID_PLACEHOLDER',
            '<HASHTAG>': 'HASHTAG_PLACEHOLDER',
            '<MENTION>': 'MENTION_PLACEHOLDER',
            '<PERCENT>': 'PERCENT_PLACEHOLDER',
            '<AGE>': 'AGE_PLACEHOLDER',
            '<TIME>': 'TIME_PLACEHOLDER',
            '<NUM>': 'NUM_PLACEHOLDER'
        }

        # Replace special tokens with placeholders
        for key, value in placeholders.items():
            self.sentences = [sentence.replace(key, value) for sentence in self.sentences]

        # Tokenize sentences
        self.sentences = [word_tokenize(sentence) for sentence in self.sentences]

        # Replace placeholders with original special tokens
        for key, value in placeholders.items():
            self.sentences = [[token if token != value else key for token in sentence] for sentence in self.sentences]

        return self.sentences

    def lowercase(self):
        self.sentences = [[word.lower() for word in sentence] for sentence in self.sentences]
    
    def remove_stop_words(self):
        stop_words = self.stop_words
        self.sentences = [[word for word in sentence if word not in stop_words] for sentence in self.sentences]

    def remove_punctuation(self):
        # Define a set of basic punctuation characters to remove
        basic_punctuation = {'.', ',', ';', ':', '!', '-', '(', ')', '[', ']', '{', '}', '"', "'"}
        self.sentences = [[word for word in sentence if word not in basic_punctuation] for sentence in self.sentences]

    # identify numbers and replace them with a special token <NUM>
    def replace_numbers(self):
        number_regex = r'\b\d+\b'  # Regular expression to match whole numbers
        self.sentences = [re.sub(number_regex, '<NUM>', sentence) for sentence in self.sentences]
    
    def replace_urls(self):
        url_regex = r'\b(?:https?://|www\.)\S+\b'
        self.sentences = [re.sub(url_regex, '<URL>', sentence) for sentence in self.sentences]
    
    def replace_emails(self):
        email_regex = r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
        self.sentences = [re.sub(email_regex, '<MAILID>', sentence, flags=re.IGNORECASE) for sentence in self.sentences]
        
    def replace_hashtags(self):
        hashtag_regex = r'\#\w+'
        self.sentences = [re.sub(hashtag_regex, '<HASHTAG>', sentence, flags=re.IGNORECASE) for sentence in self.sentences]
    
    def replace_mentions(self):
        mention_regex = r'\@\w+'
        self.sentences = [re.sub(mention_regex, '<MENTION>', sentence, flags=re.IGNORECASE) for sentence in self.sentences]
    
    def replace_percentages(self):
        percent_regex = r'\b\d+(\.\d+)?%|\b\d+(\.\d+)?\s?(percent|per cent|percentage)\b'
        self.sentences = [re.sub(percent_regex, '<PERCENT>', sentence, flags=re.IGNORECASE) for sentence in self.sentences]
    
    def replace_age_values(self):
        age_regex = r'\b\d+-year-old|\b\d+\syear(s)?\sold\b'
        self.sentences = [re.sub(age_regex, '<AGE>', sentence, flags=re.IGNORECASE) for sentence in self.sentences]
    
    def replace_time_expressions(self):
        time_regex = r'\b\d{1,2}:\d{2}\s?(AM|PM|am|pm)\b'
        self.sentences = [re.sub(time_regex, '<TIME>', sentence) for sentence in self.sentences]
    
    def replace_backslashes(self):
        backslash_regex = r'\\'
        self.sentences = [re.sub(backslash_regex, '', sentence) for sentence in self.sentences]
        backtick_regex = r'`'
        self.sentences = [re.sub(backtick_regex, '', sentence) for sentence in self.sentences]
        self.sentences = [sentence.replace("''", '') for sentence in self.sentences]
        # Regular expression to find quoted text (simple version)
        quotes_regex = r'\"[^\"]*\"'
        self.sentences = [re.sub(quotes_regex, lambda m: m.group(0)[1:-1], sentence) for sentence in self.sentences]
    
    def process(self):
        self.replace_backslashes()
        # self.replace_urls()
        # self.replace_emails()
        # self.replace_hashtags()
        # self.replace_mentions()

        self.replace_percentages()
        self.replace_age_values()
        self.replace_time_expressions()
        self.replace_numbers()

        self.word_tokenizer()

        self.lowercase()
        self.remove_stop_words()
        self.remove_punctuation()
        
        return self.sentences



# Preprocess the sentences
preprocess = Preprocess(sentences)
sentences = preprocess.process()


# Preprocess the test sentences
preprocess = Preprocess(test_sentences)
test_sentences = preprocess.process()


# Preprocess the output sentences
preprocess = Preprocess(output_sentences)
output_sentences = preprocess.process()


# print(len(sentences))
# print(sentences[:10])


class Vocabulary:
    def __init__(self, freq_threshold=1):
        # Initialize vocab with special tokens
        self.itos = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter(word for sentence in sentence_list for word in sentence)
        idx = len(self.itos)  # Start indexing for new words from here
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        return [self.stoi[word] if word in self.stoi else self.stoi["<UNK>"] for word in text]


# Create vocabulary object
vocab = Vocabulary(freq_threshold=3)
vocab.build_vocabulary(sentences)


# Check the length of the vocabulary
# print(len(vocab))
# Print some of the tokens in the vocabulary
# print(list(vocab.stoi.items())[:10])


def create_dataloader(sentences, vocab, sequence_length=50, batch_size=64):
    # Initialize lists to hold numericalized sentences (X) and targets (y)
    X = []
    y = []

    for sentence in sentences:
        # Numericalize the sentence
        numericalized = [vocab.stoi["<BOS>"]] + vocab.numericalize(sentence) + [vocab.stoi["<EOS>"]]
        # Pad or truncate the sentence to the desired sequence length
        padded = numericalized[:sequence_length] + [vocab.stoi["<PAD>"]] * max(0, sequence_length - len(numericalized))
        if len(padded) > sequence_length:
            padded = padded[:sequence_length-1] + [vocab.stoi["<EOS>"]]
        
        # Append to lists
        X.append(padded[:-1])  # Input sequence
        y.append(padded[1:])   # Target sequence (next word prediction)
    
    # Convert lists to tensors
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


# Create a dataloader
train_dataloader = create_dataloader(sentences, vocab, sequence_length=100, batch_size=32)


# ## LSTM Model


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Dropout layer before the fully connected layer
        self.fc_dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, text):
        embedded = self.embedding_dropout(self.embedding(text))
        lstm_out, _ = self.lstm(embedded)
        
        # Apply dropout to the outputs of the LSTM layer before passing to the fully connected layer
        out = self.fc(self.fc_dropout(lstm_out))
        return out


# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 64
num_layers = 2
dropout = 0.2

learning_rate = 0.001
num_epochs = 10

# Initialize the model
model = BiLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_lstm(model, train_dataloader, optimizer, criterion, device, num_epochs):
    model.to(device)  # Ensure the model is on the correct device
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.shape[-1])  # Reshape for calculating loss
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')


train_lstm(model, train_dataloader, optimizer, criterion, device, num_epochs)


# ## Perplexity


# Function to calculate perplexity
def sentence_perplexity(preprocessed_tokens, model, vocab, device):
    # Ensure the sentence starts with <BOS> and ends with <EOS>
    tokens = ['<BOS>'] + preprocessed_tokens + ['<EOS>']
    
    # Numericalize the tokens
    numericalized_tokens = [vocab.stoi.get(token, vocab.stoi['<UNK>']) for token in tokens]
    
    # Convert to a tensor and add a batch dimension
    input_tensor = torch.LongTensor(numericalized_tokens).unsqueeze(0).to(device)
    
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=2)
        
        target_probabilities = probabilities[0, range(len(numericalized_tokens)-1), numericalized_tokens[1:]]
        
        log_probabilities = torch.log2(target_probabilities)
        
        # Calculate perplexity
        perplexity = torch.pow(2, -log_probabilities.mean())

    return perplexity.item()


def average_and_median_perplexity(sentences, model, vocab, device):
    perplexities = [sentence_perplexity(sentence, model, vocab, device) for sentence in sentences]
    
    average_perplexity = sum(perplexities) / len(perplexities)
    median_perplexity = np.median(perplexities)
    
    return average_perplexity, median_perplexity


# Function to plot perplexity distribution
def plot_perplexity_distribution(sentences, model, vocab, device):
    perplexities = []
    
    for sentence in sentences:
        perplexity = sentence_perplexity(sentence, model, vocab, device)
        perplexities.append(perplexity)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(perplexities, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Sentence Perplexities')
    plt.xlabel('Perplexity')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('lstm-input:perplexity_distribution.png')
    plt.show()


# Get the average perplexity of the training sentences
train_avg, train_med = average_and_median_perplexity(output_sentences, model, vocab, device) 
print(f'Output Average Perplexity: {train_avg:.4f}')
print(f'Output Median Perplexity: {train_med:.4f}')


# Plot the perplexity distribution of the training sentences
# plot_perplexity_distribution(output_sentences, model, vocab, device)


# Get the average perplexity of the test sentences
test_avg, test_med = average_and_median_perplexity(test_sentences, model, vocab, device)
print(f'Input Average Perplexity: {test_avg:.4f}')
print(f'Input Median Perplexity: {test_med:.4f}')


# Plot the perplexity distribution of the test sentences
# plot_perplexity_distribution(test_sentences, model, vocab, device)


# ## Binary Classification


# Take a mix of output and test sentences for binary classification task
## Take 10k output sentences and 10k test sentences randomly
output_sentences_bc = random.sample(output_sentences, 10000)
test_sentences_bc = random.sample(test_sentences, 10000)

# Create a binary classification dataset
binary_classification_dataset = output_sentences_bc + test_sentences_bc
binary_classification_labels = [1] * 10000 + [0] * 10000



class PerplexityDataset(Dataset):
    def __init__(self, sentences, labels, perplexity_function):
        self.sentences = sentences
        self.labels = labels
        self.perplexity_function = perplexity_function

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        perplexity_score = self.perplexity_function(sentence)
        return torch.tensor([perplexity_score], dtype=torch.float32), torch.tensor(label, dtype=torch.long)



# Create train and test loaders for the binary classification task
train_sentences, test_sentences, train_labels, test_labels = train_test_split(binary_classification_dataset, binary_classification_labels, test_size=0.2, random_state=42)

train_dataset = PerplexityDataset(train_sentences, train_labels, lambda x: sentence_perplexity(x, model, vocab, device))
test_dataset = PerplexityDataset(test_sentences, test_labels, lambda x: sentence_perplexity(x, model, vocab, device))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class PerplexityClassifier(nn.Module):
    def __init__(self):
        super(PerplexityClassifier, self).__init__()
        self.classifier = nn.Linear(1, 2)

    def forward(self, x):
        return self.classifier(x)


# Hyperparameters
learning_rate = 0.01
num_epochs = 10

# Initialize the model
classifier = PerplexityClassifier().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)


def train_classifier(model, train_loader, criterion, optimizer, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# Call the training function
train_classifier(classifier, train_loader, criterion, optimizer, num_epochs)


def test(model, test_loader, criterion):
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs[:, 1].cpu().numpy())

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print(f"Validation Loss: {total_loss / len(test_loader):.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# Call the test function
test(classifier, test_loader, criterion)
