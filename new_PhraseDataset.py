import os
import pandas as pd
from torch.utils.data import Dataset
import torch

class PhraseDataSet(Dataset):
    def __init__(self, label_path, dir_path, tokenizer):
        self.dir_path = dir_path
        self.labels = pd.read_csv(label_path)
        self.tokenizer = tokenizer
        self.data = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        novel = self.labels.iloc[idx, 1]
        phrase = self.labels.iloc[idx, 2]
        file_name = os.path.join(self.dir_path, novel, phrase)

        with open(file_name, 'r', encoding='utf-8') as file:
            phrase = file.read()
            # transform
            phrase = phrase.strip(' ')
            phrase = self.tokenizer.tokenize(phrase)
            phrase = ['<s>'] + phrase + ['</s>']

            if (len(phrase) < 1024):
                phrase = phrase + ['<pad>'] * (1024 - len(phrase))

            item = torch.tensor(self.tokenizer.convert_tokens_to_ids(phrase))
        return item
