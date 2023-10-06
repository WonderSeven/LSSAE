import torch.nn as nn
from engine.configs import Classifers


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


@Classifers.register('nonlinear_cla')
class MultiLayerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLayerClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.linear2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.linear3 = nn.Linear(input_dim // 4, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x.view(x.size(0), -1))
        out = self.relu(out)
        out = self.relu(self.linear2(out))
        out = self.linear3(out)
        return out


@Classifers.register('linear_cla')
class SingleLayerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x.view(x.size(0), -1))
        return out
