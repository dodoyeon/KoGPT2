import argparse
from transformers import GPT2LMHeadModel, GPT2Config

from transformers import PreTrainedTokenizerFast
from MrBanana_tokenizer import MyTokenizer

import torch

device = torch.device('cpu') # 'cuda:0' imodelf torch.cuda.is_available() else
config = GPT2Config(vocab_size=52001, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
model = GPT2LMHeadModel(config)

model_dir = 'KoGPT2_weight/fine_novel_57.bin'

def encoding(text, tokenizer):
    tokens = ['<s>'] + tokenizer.tokenize(text)  # + ['</s>']
    return torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

def decoding(ids, tokenizer):
    return tokenizer.convert_ids_to_tokens(ids[0])


# def main(args):
#     device = torch.device('cpu') # 'cuda:0' if torch.cuda.is_available() else
#     config = GPT2Config(vocab_size=52001, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
#     model = GPT2LMHeadModel(config)
#
#     model_dir = args.input_path
#     model.load_state_dict(torch.load(model_dir), strict=False)
#     model.to(device)
#     model.eval()


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

    # text = args.text


    # if args.tokenizer == 'mrbnn':
    #     # MrBanana tokenizer
    #     vocab_file_path = './tokenizer/vocab.json'
    #     merge_file_path = './tokenizer/merges.txt'
    #
    #     tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
    #     bos = tokenizer.convert_tokens_to_ids('<s>') # 0
    #     eos = tokenizer.convert_tokens_to_ids('</s>') # 2.....
    #     pad = tokenizer.convert_tokens_to_ids('<pad>') # 1
    #     unk = tokenizer.convert_tokens_to_ids('<unk>') # 3
    #
    #     def add_special_tokens_(model, tokenizer):
    #         orig_num_tokens = tokenizer.get_vocab_size()
    #         model.resize_token_embeddings(new_num_tokens=orig_num_tokens + 1)
    #
    #     add_special_tokens_(model, tokenizer)
    #     input_ids = encoding(text, tokenizer)
    # elif args.tokenizer == 'kogpt2':
    #     # SKT pre-trained tokenizer
    #     tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
    #                                                         bos_token='<s>', eos_token='</s>', unk_token='<unk>',
    #                                                         pad_token='<pad>', mask_token='<mask>')
    #     pad = tokenizer.pad_token_id
    #     bos = tokenizer.bos_token_id
    #     eos = tokenizer.eos_token_id
    #     input_ids = torch.tensor([tokenizer.encode(text)])
    #
    # sample_outputs = model.generate(
    #     input_ids,
    #     do_sample=True,
    #     max_length=1024,
    #     no_repeat_ngram_size=2,
    #     top_k=50,
    #     top_p=0.95,
    #     pad_token_id=pad,
    #     bos_token_id=bos,
    #     eos_token_id=eos,
    #     early_stopping=True
    #     # bad_words_ids=[unk]
    # )
    #
    # if args.tokenizer == 'mrbnn':
    #     print(decoding(sample_outputs.tolist(), tokenizer))
    # elif args.tokenizer == 'kogpt2':
    #     print(tokenizer.decode(sample_outputs[0,:].tolist()))

def generator(input):
    input_ids = encoding(input, tokenizer)

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
    return decoding(sample_outputs.tolist(), tokenizer)
    
generator('어느 화창한 날 동산위에 착륙한 우주선에서 외계인이 내려와')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', '-i', default='KoGPT2_weight/fine_novel_jw_0.bin', type=str,
#                         dest='input_path', help='Input weight file')
#     parser.add_argument('--text', '-t', default='그 외계인은 내가 좋다고 말했다.', type=str,
#                         dest='text', help='Text for test')
#     parser.add_argument('--tokenizer', '-token', default='kogpt2', type=str,
#                         dest='tokenizer', help='Type of tokenizer(kogpt2, mrbnn)')
#     args = parser.parse_args()
#
#     main(args)

