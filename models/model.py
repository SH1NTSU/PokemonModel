import torch.nn as nn
import timm

class PokemonClassifier(nn.Module):
    def __init__(self, num_classes=102):
        super(PokemonClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b2', pretrained=True)
        self.base_model.classifier = nn.Identity()  # Remove the default classifier
        self.classifier = nn.Linear(self.base_model.num_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

