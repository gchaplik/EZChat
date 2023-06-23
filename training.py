import json
from EZChatP1 import token,BOW,LS

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

