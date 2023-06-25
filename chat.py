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