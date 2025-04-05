# Cox + Transformer
import torch.nn as nn

class SurvivalTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=2), num_layers=1)
        self.risk = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.risk(x.mean(dim=1))