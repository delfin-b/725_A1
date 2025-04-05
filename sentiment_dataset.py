import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import re

class SentimentDataset(Dataset):
    def __init__(self, csv_file, meta_file, max_length=256):
        self.data = pd.read_csv(csv_file)
        # Load vocabulary from the meta file (assumes meta contains a 'stoi' mapping)
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        self.stoi = meta['stoi']
        self.max_length = max_length
        # Map sentiment strings to integer labels
        self.sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
    def preprocess_text(self, text):
        # Basic cleaning: lowercasing and whitespace normalization (punctuation already removed in preprocessing)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text):
        # Here we split on whitespace; you could also apply a more sophisticated tokenization if desired.
        tokens = text.split()
        # Convert tokens to indices, using 0 for any unknown tokens
        return [self.stoi.get(token, 0) for token in tokens]
    
    def pad_or_truncate(self, token_ids):
        # Ensure the sequence has exactly max_length tokens
        if len(token_ids) > self.max_length:
            return token_ids[:self.max_length]
        else:
            return token_ids + [0] * (self.max_length - len(token_ids))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = self.preprocess_text(row['clean_conversation'])
        token_ids = self.tokenize(text)
        token_ids = self.pad_or_truncate(token_ids)
        label = self.sentiment_map[row['customer_sentiment']]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
