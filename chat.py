import random 
import json
import torch
from ml_model import NeuralNet
from EZChatP1 import BOW, token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# open saved data model
with open('data_tags.json', 'r') as f:
    intent = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['collected_words']
tags = data['collected_tags']
model_state = data['model_state']

# load learned params 
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)

# set model to eval mode
model.eval()

bot_name = "OniBot"
print("Lets chat! type 'quit' to exit")

while True:
    sentence = input('You ')
    if sentence == 'quit':
        break
    
    sentence = token(sentence)
    X = BOW(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, prediction = torch.max(output, dim = 1)
    tag = tags[prediction.item()]
    
    probability = torch.softmax(output, dim = 1)
    probability = probability[0][prediction.item()]
        
    if probability.item() > 0.75:    
        for i in intent['intents']:
            if tag == i["tag"]:
                print(f'{bot_name}: {random.choice(i["response"])}')

    else:
        print(f'{bot_name}: I cannot understand')
