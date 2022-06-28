import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.configs import Classifers


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# @Classifers.register('toy_linear_cla')
# class ToyLinearClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ToyLinearClassifier, self).__init__()
#         no_bn = True
#         self.fc3 = nn.Linear(input_dim, input_dim)
#         self.bn3 = nn.BatchNorm1d(input_dim)
#         self.fc4 = nn.Linear(input_dim, input_dim)
#         self.bn4 = nn.BatchNorm1d(input_dim)
#         self.fc_final = nn.Linear(input_dim, output_dim)
#         self.relu = nn.ReLU()
#
#         if no_bn:
#             self.bn3 = Identity()
#             self.bn4 = Identity()
#
#     def forward(self, x, return_softmax=False):
#         out = self.relu(self.bn3(self.fc3(x)))
#         out = self.relu(self.bn4(self.fc4(out)))
#         out = self.fc_final(out)
#         out_softmax = F.softmax(out, dim=1)
#         out = F.log_softmax(out, dim=1)
#
#         if return_softmax:
#             return out, out_softmax
#         else:
#             return out


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
