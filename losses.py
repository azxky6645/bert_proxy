import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses


def binarize(T, n_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, n_classes))
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(nn.Module):
    def __init__(self, label_emb, n_classes, hidden_size, mrg=0.2, alpha=64):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        if label_emb == None:
            self.proxies = torch.nn.Parameter(torch.randn(n_classes, hidden_size).cuda())
            nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        else:
            self.proxies = torch.nn.Parameter(label_emb)


        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, n_classes=self.n_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.n_classes
        loss = pos_term + neg_term

        return loss, P