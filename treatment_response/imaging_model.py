import torch.nn as nn

class ImagingResponseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16*6*6, 2)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))