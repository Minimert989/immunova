# Multimodal fusion
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_branch = nn.Linear(128, 64)
        self.gen_branch = nn.Linear(128, 64)
        self.fusion = nn.Linear(128, 2)

    def forward(self, img_feat, gen_feat):
        i = self.img_branch(img_feat)
        g = self.gen_branch(gen_feat)
        combined = torch.cat([i, g], dim=1)
        return self.fusion(combined)