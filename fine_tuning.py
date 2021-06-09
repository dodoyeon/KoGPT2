# NarrativeKoGPT2 ref : Fine tuning
from PhraseDataset import PhraseDataSet
import os
import random
import torch
import math
import argparse

from MrBanana_tokenizer import MyTokenizer

from torch.utils.data import DataLoader

import kss
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm
from tensorboardX import SummaryWriter

from NovelDataset import NovelDataSet

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config['input_path'] is not None:
        model = GPT2LMHeadModel(config=GPT2Config(vocab_size=52000))
        model.load_state_dict(torch.load(config['input_path']), strict=False)
    else:
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2') #
    model.to(device) # 1510-11MiB

    vocab_file_path = './tokenizer/vocab.json'
    merge_file_path = './tokenizer/merges.txt'

    # ATTR_TO_SPECIAL_TOKEN = ['<s>','</s>']
    #
    def add_special_tokens_(model, tokenizer):
        orig_num_tokens = tokenizer.get_vocab_size()
        # tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        # num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
        # new_num_tokens=orig_num_tokens + num_added_tokens + 1
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + 1)

    if config['tokenizer'] == 'kogpt2':
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')
    elif config['tokenizer'] == 'mrbnn':
        tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
        add_special_tokens_(model, tokenizer)
    else:
        raise ValueError('Not defined tokenizer.')

    learning_rate = config['lr']
    epochs = config['epoch']
    batch_size = config['batch_size'] # 4

    if config['data_type'] == 'sentence':
        # Sentence Dataset
        file_path = 'GPT2_dataset/'
        dataset = NovelDataSet(file_path, tokenizer)
    elif config['data_type'] == 'phrase':
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

    loss_dir = config['loss_dir']
    if not os.path.isdir(loss_dir):
        os.mkdir(loss_dir)
    loss_file = os.path.join(loss_dir, 'loss.txt')

    weight_dir = config['weight_dir']
    if not os.path.isdir(weight_dir):
        os.mkdir(weight_dir)

    summary = SummaryWriter()
    prefix = config['prefix_weight']
    output_freq = config['output_freq']
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

            if (count+1) % output_freq == 0:
                print('epoch {0} train_iteration {1} | loss = {2:.5f} avg_loss = {3:.5f}'.format(epoch, count, loss, avg_loss[0]/avg_loss[1]))
                l = 'epoch' + str(epoch) + 'train_iteration' + str(count) + ' | loss: ' + str(loss) + 'avg_loss: ' + str(avg_loss)
                with open(loss_file, 'a') as f:
                    f.write(l)
                    f.write('\n')
                summary.add_scalar('loss/avg_loss', avg_loss[0] / avg_loss[1], total_count)
                summary.add_scalar('loss/loss', loss, total_count)
            count += 1
            total_count += 1

        torch.save({'epoch': epoch, 'model state_dict': model.state_dict()}, weight_dir + prefix + str(epoch) + '.bin')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        dest='epoch', help='training epoch')
    parser.add_argument('--learning-rate', '-lr', default=1e-5, type=float,
                        dest='lr', help='training learning rate')
    parser.add_argument('--batch-size', '-bs', default=2, type=int,
                        dest='batch_size', help='training batch size')
    parser.add_argument('--loss-dir', '-ld', default='loss/', type=str,
                        dest='loss_dir', help='Path to save log for training loss')
    parser.add_argument('--weight-dir', '-wd', default='weight/', type=str,
                        dest='weight_dir', help='Path to save weight of model')
    parser.add_argument('--weight-file', '-wf', default='fine_novel_', type=str,
                        dest='prefix_weight', help='Prefix for weight files')
    parser.add_argument('--data-type', '-dt', default='phrase', type=str,
                        dest='data_type', help='Type of Dataset(sentence, phrase)')
    parser.add_argument('--output-frequency', '-of', default=40, type=int,
                        dest='output_freq', help='Frequency of results of loss')
    parser.add_argument('--tokenizer', '-t', default='kogpt2', type=str,
                        dest='tokenizer', help='Type of tokenizer(kogpt2, mrbnn)')
    parser.add_argument('--input-weight', '-i', default=None, type=str,
                        dest='input_path', help='Pre-trained weight')
    args = parser.parse_args()

    config = {
        'epoch': args.epoch,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'loss_dir': args.loss_dir,
        'weight_dir': args.weight_dir,
        'prefix_weight': args.prefix_weight,
        'data_type': args.data_type,
        'output_freq': args.output_freq,
        'tokenizer': args.tokenizer,
        'input_path': args.input_path
    }

    main(config)
