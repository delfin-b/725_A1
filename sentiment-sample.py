"""
Sample from a trained sentiment analysis model

This script loads a GPT model modified for sentiment classification,
preprocesses a customer conversation, and outputs the predicted sentiment.
"""

import os
import pickle
import torch
import re
import pandas as pd
from model import GPTConfig, GPT  # modified GPT model with a classification head

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load meta information from preprocessing step (vocabulary mappings)
meta_path = 'data/customer_service/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']  # Mapping from tokens to indices

# Define a simple tokenizer matching the training preprocessing.
def tokenize(text):
    # Lowercase and strip extra whitespace
    text = text.lower().strip()
    # Split text on whitespace (can bbe modified if you used a different tokenization during training)
    tokens = text.split()
    # Convert tokens to indices; unknown tokens default to 0.
    return [stoi.get(token, 0) for token in tokens]

# Define a function to pad or truncate a sequence to a fixed length.
def pad_sequence(token_ids, max_length=256):
    if len(token_ids) >= max_length:
        return token_ids[:max_length]
    else:
        return token_ids + [0] * (max_length - len(token_ids))

# Preprocessing function matching our training script.
def preprocess_text(text):
    # Convert text to lowercase, normalize whitespace, and remove punctuation.
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


#####################
# Model configuration (aligned with the configuration used during training)

block_size = 256
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2
bias = False
vocab_size = meta['vocab_size']

config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias
)

# Initialize the modified GPT model for sentiment analysis (3 classes: negative, neutral, positive)
model = GPT(config, num_classes=3)
ckpt_path = os.path.join('out-sentiment-small', 'ckpt.pt') ### pls check the path from -> config/sentiment_class.py


# Load the checkpoint
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()  # Set model to evaluation mode

# Define a mapping from output indices to sentiment labels.
sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

test_csv = 'data/customer_service/test.csv'  # Path to the test CSV file
test_df = pd.read_csv(test_csv)

# use random sample
sample_row = test_df.sample(n=1).iloc[0]

input_text = sample_row['conversation']
print(f"Input Text: {input_text}")
print("************************")

# prep, tokenize and pad
clean_text = preprocess_text(input_text)
token_ids = tokenize(clean_text)
token_ids = pad_sequence(token_ids, max_length=block_size)
x= torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device) 

#run inference for pred
with torch.no_grad():
    logits, _= model(x, targets=None)
    prediction = torch.argmax(logits, dim=-1).item()

print(f"Predicted Sentiment: {sentiment_map[prediction]}")