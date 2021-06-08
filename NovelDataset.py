import os

from torch.utils.data import Dataset
import torch

class NovelDataSet(Dataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

    def __len__(self):
        data_list = os.listdir(self.file_path)
        return len(data_list)

    def __getitem__(self, index):
        file_name = 'novel_sentence'+str(index)+'.txt'
        text_path = os.path.join(self.file_path, file_name)
        with open(text_path, 'r', encoding = 'utf-8') as file:
            line = file.read()
            
            # transform
            line = self.tokenizer.tokenize(line)
            line = ['<s>'] + line + ['</s>']
            
            line = line + ['<pad>'] * (1024 - len(line))
            item = torch.tensor(self.tokenizer.convert_tokens_to_ids(line))
            # item = tokenizer.convert_tokens_to_ids(line)
        return item
