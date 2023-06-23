import json
from EZChatP1 import token,BOW,LS
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

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


bs=8
ds=CDS()
tLoader = DataLoader(dataset=ds,batch_size=bs,shuffle=True,num_workers=2)