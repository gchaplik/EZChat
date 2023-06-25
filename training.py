import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from EZChatP1 import BOW, token, LS
from ml_model import NeuralNet

with open('data_tags.json', 'r') as f:
    intents = json.load(f)

allWords = []
tags = []
xy = []
xtrain = []
ytrain = []
ignore=["?",".","/",",","+","-","(",")"]
for i in intents['intents']:
    tag = i['tag']
    tags.append(tag)
    for p in i['patterns']:
        w = token(p)
        allWords.extend(w)
        xy.append((w, tag))



all_words = [LS(w) for w in allWords if w not in ignore]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)



for (pattern_sentence, tag) in xy:

    bag = BOW(pattern_sentence, all_words)
    xtrain.append(bag)

    label = tags.index(tag)
    ytrain.append(label)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(xtrain[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(xtrain)
        self.x_data = xtrain
        self.y_data = ytrain
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)

        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

# Define model params
data_model = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "collected_words": all_words,
    "collected_tags": tags
}

# Save model to pickled file
FILE = "data.pth"
torch.save(data_model, FILE)
print(f'Training Complete. File saved to {FILE}')