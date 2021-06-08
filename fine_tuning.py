# NarrativeKoGPT2 ref : Fine tuning
import os
import random
import torch

from MrBanana_tokenizer import MyTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import kss
from transformers import GPT2LMHeadModel #,AdamW #, PreTrainedTokenizerFast
from tqdm import tqdm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2') #
model.config
model.to(device) # 1510-11MiB

vocab_file_path = './tokenizer/vocab.json'
merge_file_path = './tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
# ATTR_TO_SPECIAL_TOKEN = ['<s>','</s>']
#
def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    # tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    # num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + 1) # new_num_tokens=orig_num_tokens + num_added_tokens + 1

# add_special_tokens_(model, tokenizer)

class NovelDataSet(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

    # def load_data(self, data_path): # , data_path
    #     # with open("GPT2_dataset/1twilight.txt", 'r', encoding='utf-8') as file:
    #     file = open(data_path, 'r', encoding='utf-8')
    #     text = file.read()
    #     file.close()
    #     # text = text.replace("\"", "\n")
    #     text = text.replace("\n", "")
    #
    #     split_list = kss.split_sentences(text)
    #
    #     for line in tqdm(split_list):
    #         tok = tokenizer.tokenize(line)
    #         tokenized_line = ['<s>'] + tok + ['</s>']
    #         if len(tokenized_line) < 1024:
    #             padded_data = tokenized_line + ['<pad>'] * (1024 - len(tokenized_line))
    #             self.data.append(torch.tensor(tokenizer.convert_tokens_to_ids(padded_data)).unsqueeze(0))
    #         else:
    #             sent = []
    #             flag = False
    #             for e in tok:
    #                 sent.append(e)
    #                 if not flag and (e == '"' or e == '“'):
    #                     flag = True
    #                 elif flag and (e == '"' or e == '”' or e == '’' or e == "'"):
    #                     flag = False
    #                     tokenized_line = ['<s>'] + sent + ['</s>']
    #                     padded_data = tokenized_line + ['<pad>'] * (1024 - len(tokenized_line))
    #                     self.data.append(torch.tensor(tokenizer.convert_tokens_to_ids(padded_data)).unsqueeze(0))
    #                     sent = []
    #                 elif not flag and e == '.':
    #                     tokenized_line = ['<s>'] + sent + ['</s>']
    #                     padded_data = tokenized_line + ['<pad>'] * (1024 - len(tokenized_line))
    #                     self.data.append(torch.tensor(tokenizer.convert_tokens_to_ids(padded_data)).unsqueeze(0))
    #                     sent = []
    #                 elif flag and len(sent) == 1000:
    #                     flag = True


    # def load_total(self):
    #     data_list = os.listdir(self.file_path)
    #     for path in data_list:
    #         data_path = self.file_path + '/' + path
    #         self.load_data(data_path)

    def __len__(self):
        data_list = os.listdir(self.file_path)
        return len(data_list)

    def __getitem__(self, index):
        file_name = 'novel_sentence'+str(index)+'.txt'
        text_path = os.path.join(self.file_path, file_name)
        with open(text_path, 'r', encoding = 'utf-8') as file:
            line = file.read()
            
            # transform
            line = tokenizer.tokenize(line)
            line = ['<s>'] + line + ['</s>']
            
            line = line + ['<pad>'] * (1024 - len(line))
            item = torch.tensor(tokenizer.convert_tokens_to_ids(line))
            # item = tokenizer.convert_tokens_to_ids(line)
        return item

# class NovelDataLoader(DataLoader):

learning_rate = 1e-5
epochs = 20
batch_size = 2 # 4

file_path = 'GPT2_dataset'
dataset = NovelDataSet(file_path)
# dataset.load_total()
novel_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()

# criterion = torch.nn.CrossEntropyLoss() -> 필요없음
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
# optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=True)
count = 0
avg_loss = (0.0, 0.0)

loss_file_dir = 'KoGPT-2Model/loss.txt'
# loss_file = open(loss_file_dir, 'a')

for epoch in tqdm(range(epochs)):
    count = 0
    for batch in novel_dataloader:
        # print('start training..')
        optimizer.zero_grad()
        
        # batch = torch.stack(batch)
        batch = batch.transpose(1,0) # batch (1024, 2)
        batch = batch.to(device)

        outputs = model(batch, labels=batch)
        loss, logits = outputs[:2] # ??
        loss.to(device)
        loss.backward()
        avg_loss = (avg_loss[0]*0.99 + loss, avg_loss[1]*0.99 + 1.0)
        optimizer.step()

        if (count+1) % 200 == 0:
            print('epoch {0} train_iteration {1} | loss = {2:.5f} avg_loss = {3:.5f}'.format(epoch, count, loss, avg_loss[0]/avg_loss[1]))
            with open(loss_file_dir, 'a') as loss_file:
                l = 'epoch' + str(epoch) + 'train_iteration' + str(count) + ' | loss: ' + str(loss) + 'avg_loss: ' + str(avg_loss)
                loss_file.write(l)
                loss_file.write('\n')
        count += 1

    torch.save({'epoch': epoch, 'model state_dict': model.state_dict()}, 'KoGPT2_weight/fine_novel_jw_' + str(epoch) + '.bin')
