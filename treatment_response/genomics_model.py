# Genomics model using Transformer
import torch.nn as nn

class GenomicsTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4), num_layers=2)
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))