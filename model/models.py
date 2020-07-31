from base import BaseModel
import torch
from model.module import VGG, CraftDecoder


class CraftModel(BaseModel):
    def __init__(self, pretrained=True, freeze=True):
        super(CraftModel, self).__init__()
        self.encoder = VGG(pretrained=pretrained, freeze=freeze)
        self.decoder = CraftDecoder()

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)

        return out


if __name__ == '__main__':
    model = CraftModel()
    output, _ = model(torch.randn(1, 3, 512, 512))
    print(output.shape)
