# NarrativeKoGPT2 ref : Fine tuning
from PhraseDataset import PhraseDataSet
import os
import random
import torch

from MrBanana_tokenizer import MyTokenizer

from torch.utils.data import DataLoader

import kss
from transformers import GPT2Config, GPT2LMHeadModel #,AdamW #, PreTrainedTokenizerFast
from tqdm import tqdm
from tensorboardX import SummaryWriter

from NovelDataset import NovelDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GPT2LMHeadModel(config=GPT2Config(vocab_size=52000))
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2') #
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

add_special_tokens_(model, tokenizer)

learning_rate = 1e-5
epochs = 20
batch_size = 2 # 4

# Sentence Dataset
# file_path = 'GPT2_dataset/'
# dataset = NovelDataSet(file_path, tokenizer)

# Phrase Dataset
file_path = 'phraseDataset/'
label_path = file_path + 'label.csv'
dataset = PhraseDataSet(label_path, file_path, tokenizer)

novel_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
# optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=True)
count = 0
avg_loss = (0.0, 0.0)

loss_file_dir = 'KoGPT-2Model/'
if not os.path.isdir(loss_file_dir):
    os.mkdir(loss_file_dir)
loss_file = os.path.join(loss_file_dir, 'loss.txt')

weight_dir = 'KoGPT2_weight/'
if not os.path.isdir(weight_dir):
    os.mkdir(weight_dir)

summary = SummaryWriter()

total_count = 0
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
            l = 'epoch' + str(epoch) + 'train_iteration' + str(count) + ' | loss: ' + str(loss) + 'avg_loss: ' + str(avg_loss)
            with open(loss_file, 'a') as f:
                f.write(l)
                f.write('\n')
            summary.add_scalar('loss/avg_loss', avg_loss[0] / avg_loss[1], total_count)
            summary.add_scalar('loss/loss', loss, total_count)
        count += 1
        total_count += 1

    torch.save({'epoch': epoch, 'model state_dict': model.state_dict()}, weight_dir + 'fine_novel_jw_' + str(epoch) + '.bin')
