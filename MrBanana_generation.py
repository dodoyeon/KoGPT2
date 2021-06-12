from transformers import GPT2LMHeadModel, GPT2Config

from transformers import PreTrainedTokenizerFast
from MrBanana_tokenizer import MyTokenizer

import torch

from flask import Flask, request
from flask_restx import Resource, Api, reqparse, Namespace

device = torch.device('cpu') # 'cuda:0' imodelf torch.cuda.is_available() else
config = GPT2Config(vocab_size=52001, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
model = GPT2LMHeadModel(config)

model_dir = 'KoGPT2_weight/fine_novel_57.bin'

model.to(torch.device('cpu'))
model.load_state_dict(torch.load(model_dir), strict=False)
model.eval()

# model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
# model.config
# model.to(device)

# MrBanana tokenizer
vocab_file_path = './tokenizer/vocab.json'
merge_file_path = './tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
bos = tokenizer.convert_tokens_to_ids('<s>') # 0
eos = tokenizer.convert_tokens_to_ids('</s>') # 2
pad = tokenizer.convert_tokens_to_ids('<pad>') # 1
unk = tokenizer.convert_tokens_to_ids('<unk>') # 3

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + 1)

add_special_tokens_(model, tokenizer)

# SKT pre-trained tokenizer
# tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
#                                                     bos_token='<s>', eos_token='</s>', unk_token='<unk>',
#                                                     pad_token='<pad>', mask_token='<mask>')

def encoding(text):
    tokens = ['<s>'] + tokenizer.tokenize(text)# + ['</s>']
    return torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

def decoding(ids):
    return tokenizer.convert_ids_to_tokens(ids[0])

def generator(input):
    input_ids = encoding(input)

    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=128,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        pad_token_id=pad,
        bos_token_id=bos,
        eos_token_id=eos,
        early_stopping=True
        # bad_words_ids=[unk]
    )
    return decoding(sample_outputs.tolist())
    
# generator('어느 화창한 날 동산위에 착륙한 우주선에서 외계인이 내려와')

# novel = {}
# count = 1
#
# Relay = Namespace('Relay')
#
# parser = reqparse.RequestParser()
# parser.add_argument('name', type=str, help= 'user name')
# parser.add_argument('input sentence', type = str, help= 'input korean sentence')
# args = parser.parse_args()
#
# @Relay.route('/relay_novel_generator')
# class RelayGen(Resource):
#     def post(self):
#         global novel
#         global count
#         idx = count
#         count += 2
#
#         novel[idx] = request.json.get(args['input sentence'])
#         out = generator(args['input sentence'])
#         novel[idx+1] = request.json.get(out)
#         output = {'AI responce': out}
#
#         return output, 200