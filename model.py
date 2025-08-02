import torch
import torch.nn as nn

class simpleNN(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 128, num_classes = 10):
        super(simpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)               # convert this into 784 dimension tensor

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

        