import json
from utils.nltk_utils import bag_of_word, stem, tokenize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('src/intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        
        
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_word(pattern_sentence, all_words)
    
    X_train.append(tag)
    
    label = tags.index(tag)
    print(label)
    y_train.append(label) 
    
X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

# Hyper parameters
batch_size = 8

dataset = ChatDataset()
train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          )
