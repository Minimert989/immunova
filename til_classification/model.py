import torchvision.models as models

class TILClassifier:
    def __init__(self, model_type='resnet'):
        if model_type == 'resnet':
            self.model = models.resnet18(pretrained=True)
        else:
            self.model = models.efficientnet_b0(pretrained=True)