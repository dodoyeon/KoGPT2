import random
import torch
from torch.utils.data import DataLoader
from MrBanana_tokenizer import MyTokenizer
from transformers import GPT2LMHeadModel, GPT2Config
import Narrative_sampling as sampling
import kss

# Model Definition
config = GPT2Config(vocab_size=52001, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
model = GPT2LMHeadModel(config)

model_dir = 'KoGPT2_weight/fine_novel.bin'
model.load_state_dict(torch.load(model_dir), strict=False)
model.to(torch.device('cpu'))

# Tokenizer Definition
vocab_file_path = './tokenizer/vocab.json'
merge_file_path = './tokenizer/merges.txt'
tokenizer = MyTokenizer(vocab_file_path, merge_file_path)

bos = tokenizer.convert_tokens_to_ids('<s>')
eos = tokenizer.convert_tokens_to_ids('</s>')
pad = tokenizer.convert_tokens_to_ids('<pad>')
unk = tokenizer.convert_tokens_to_ids('<unk>')

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + 1)

add_special_tokens_(model, tokenizer)

sent = input('문장 입력: ')

toked = tok(sent)
count = 0
output_size = 200 # 출력하고자 하는 토큰 갯수

while 1:
  input_ids = torch.tensor(['<s>'] + toked).unsqueeze(0)
  predicts = model(input_ids)
  pred = predicts[0]

  last_pred = pred.squeeze()[-1]
  # top_p 샘플링 방법
  # sampling.py를 통해 random, top-k, top-p 선택 가능.
  # gen = sampling.top_p(last_pred, vocab, 0.98)
  gen = sampling.top_k(last_pred, vocab, 5)

  if count>output_size:
    sent += gen.replace('▁', ' ')
    toked = tok(sent)
    count =0
    break
  sent += gen.replace('▁', ' ')
  toked = tok(sent)
  count += 1

for s in kss.split_sentences(sent):
    print(s)

