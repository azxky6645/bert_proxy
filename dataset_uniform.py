import os
import re
import html

import pandas as pd
from torch.utils.data import Dataset, IterableDataset
import torch
import time

# LABEL_DICT ={       # emory_nlp_classes
#     'Neutral': 0,
#     'Scared': 1,
#     'Joyful': 2,
#     'Peaceful': 3,
#     'Mad': 4,
#     'Sad': 5,
#     'Powerful': 6
# }

LABEL_DICT ={       # MELD, neutral class delete
    'joy': 0,
    'surprise': 1,
    'anger': 2,
    'sadness': 3,
    'disgust': 4,
    'fear': 5,
    'neutral': 6
}


class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        df = pd.read_csv(data_path, sep='\t') # tsv file required
        self.data = df.reset_index(drop=True)
        self.text = self.data['Utterance']
        self.label = [LABEL_DICT[e] for e in self.data['Emotion']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.normalize_string(self.text[idx])
        token = self.tokenizer(text,
                               padding='max_length',
                               max_length=self.max_seq_len
                               )
        token = token.convert_to_tensors(prepend_batch_axis=False, tensor_type='pt')
        label = self.label[idx]

        #label = self.normalize_string(self.tokenizer.tokeinze(self.label[idx]))
        return token, label

    @staticmethod
    def normalize_string(s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s



# class MyIterableDataset(IterableDataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __iter__(self):
#         for x in self.data:
#             worker = torch.utils.data.get_worker_info()
#             worker_id = worker.id if worker is not None else -1
#
#             start = time.time()
#             time.sleep(0.1)
#             end = time.time()
#
#             yield x, worker_id, start, end


# Iterable dataset의 경우 직접 worker 별로 일을 재분배 해야함
def worker_init_fn():
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) // worker_info.num_workers
    dataset.data = dataset.data[worker_id * split_size: (worker_id + 1) * split_size]
