import torch
import torch.nn as nn

# Define neutral network and params
# input size, number of classes should be static but layer size is dynamic
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes):   
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer_size)
        self.layer2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.layer3 = nn.Linear(hidden_layer_size, num_classes)
        # Define activation function
        self.activation = nn.ReLU()
    
    def fwd(self, x):
        output = self.layer1(x)
        output = self.activation(output)
        output = self.layer2(x)
        output = self.activation(output)
        output = self.layer3(x)
        output = self.activation(output)
        
        # no activation function and no softmax
        return output