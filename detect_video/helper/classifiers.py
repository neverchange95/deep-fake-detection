from functools import partial

import torch
import timm
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

encoder_params = {
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(timm.create_model, 'tf_efficientnet_b7.ns_jft_in1k', pretrained=True, drop_path_rate=0.2)
    },
}

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
