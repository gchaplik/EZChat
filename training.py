import json
from EZChatP1 import token,BOW,LS
import numpy as np
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

ignore=["?",".","/",",","+","-","(",")","\n"]

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