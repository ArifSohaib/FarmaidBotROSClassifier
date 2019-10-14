from torch.nn import Sequential, Flatten, BatchNorm1d, Dropout, ReLU, Linear, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Module
import torch
import torch.nn as nn
from torchvision import models

class AdaptiveConcatPool2d(Module):

    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz):
        super(AdaptiveConcatPool2d,self).__init__()

        "Output will be 2*sz or 2 if sz is None"

        self.output_size = sz or 1

        self.ap = nn.AdaptiveAvgPool2d(self.output_size)

        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def get_model():
    model_head = torch.nn.Sequential(

            AdaptiveConcatPool2d(1),

            Flatten(),

            BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            Dropout(p=0.25, inplace=False),

            Linear(in_features=1024, out_features=512, bias=True),

            ReLU(inplace=True),

            BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            Dropout(p=0.5, inplace=False),

            Linear(in_features=512, out_features=5, bias=True))

    resnet34 = models.resnet34()
    model_body = nn.Sequential(*list(resnet34.children()))[:-2]
    pytorch_model = torch.nn.Sequential(model_body, model_head)
    return pytorch_model

def main():
    pytorch_model = get_model()
    pytorch_model.load_state_dict(torch.load("../models/farmaid_model_pytorch"))
    pytorch_model.eval()

if __name__ == '__main__':
    main()
