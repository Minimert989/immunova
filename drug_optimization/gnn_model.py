# GNN for drug design
import torch.nn as nn

class DrugGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        return self.fc(x)