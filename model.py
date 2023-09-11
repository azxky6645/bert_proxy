import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForSequenceClassification, AutoTokenizer


class BertProxy(nn.Module):
    def __init__(self, config):
        super(BertProxy, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                               attention_probs_dropout_prob=0.1,
                                               hidden_dropout_prob=0.1)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(config.dropout),
        #     nn.Linear(config.hidden_size, config.n_classes)
        # )
    def forward(self, x):
        pooled_output = self.model(**x).pooler_output
        return pooled_output

class BertCross(nn.Module):
    def __init__(self, config):
        super(BertCross, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                               attention_probs_dropout_prob=0.1,
                                               hidden_dropout_prob=0.1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.n_classes)
        )

    def forward(self, x):
        pooled_output = self.model(**x).pooler_output
        output = self.classifier(pooled_output)
        return output
