import torch
import torch.nn as nn
import torch.nn.functional as F


class DAttention(nn.Module):
    def __init__(self, n_classes, dropout, act, n_features=1024):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(n_features, 512)]

        if act.lower() == "gelu":
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

    def forward(self, x, return_attn=False, return_embedding=False):
        feature = self.feature(x)   # [1, N, n_features] -> [1, N, L]
        feature = feature.squeeze() # [1, N, L] -> [N, L]
        A = self.attention(feature)    # [N, L] -> [N, K] = [N, 1]
        A = torch.transpose(A, -1, -2)  # KxN = [1, N]
        A = F.softmax(A, dim=-1)  # softmax over N, [1, N]
        M = torch.mm(A, feature)  # [1, N] x [N, L] = [1, L]
        logits = self.classifier(M) # [1, L] -> [1, n_classes]
        if return_embedding:
            return logits, M.detach().cpu().squeeze(0).numpy()  # [L,]
        elif return_attn:
            return logits, A
        else:
            return logits
