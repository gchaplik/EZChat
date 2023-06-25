import json
from EZChatP1 import token,BOW,LS
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from ml_model import NeuralNetwork

with open("ourjsonFile",'r') as f:
    intents = json.load(f)

allWords=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=token(pattern)
        allWords.extend(w)
        xy.append((w,tag))

ignore=["?",".","/",",","+","-","(",")"]

allWords=[LS(w) for w in allWords if w not in ignore]
allWords=sorted(set(allWords))
tags=sorted(set(tags))
xtrain=[]
ytrain=[]
for(patternS,tag) in xy:
    bag=BOW(patternS,allWords)
    xtrain.append(bag)
    l = tags.index(tag)
    ytrain.append(l)
xtrain=np.array(xtrain)
ytrain=np.array(ytrain)

class CDS(ds):
    def __init__(self):
        self.numSamples=len(xtrain)
        self.xData=xtrain
        self.yData=ytrain
    def __get__(self,index):
        return self.xData[index],self.yData[index]
    def __len__(self):
        return self.numSamples

# params
bs=8

# let the input size based off the 1st bag of words
hidden_layers = 8
output_size = len(tags)
input_size = len(xtrain[0])
learn_rate = 0.001
num_epochs = 1000 # 1000 iterations

ds=CDS()
tLoader = DataLoader(dataset=ds,batch_size=bs,shuffle=True,num_workers=2)

# define model within training set
user_device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = NeuralNetwork(input_size, hidden_layer_size, num_classes)

# loss and optimizer parameters
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)

for epoch in range(num_epochs):
    for (words, labels) in tLoader:
        words = words.to(user_device)
        labels = labels.to(user_device)
        
        # fwd activation
        outputs = model(word)
        loss = criteria(outputs, labels)
        
        # optimizer and backtrack
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # debug output
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, loss = {loss.item()}') 

print(f'Final model loss: {loss.item()}')
        