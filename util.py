from transformers import GPT2LMHeadModel, GPT2Config
from transformers import PreTrainedTokenizerFast
from MrBanana_tokenizer import MyTokenizer
from os import listdir, mkdir
from os.path import isfile, isdir, join
import torch

# Generation code
def encoding(text, tokenizer):
    tokens = ['<s>'] + tokenizer.tokenize(text)# + ['</s>']
    return torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

def decoding(ids, tokenizer):
    return tokenizer.decode(ids[0])
    # return tokenizer.convert_ids_to_tokens(ids[0])

# Tokenizer
def add_special_tokens_(model, tokenizer, SPECIAL_TOKEN):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(SPECIAL_TOKEN)
    num_add_tokens = len(SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_add_tokens + 1)

# Data Distributor
def save_phrase(label_novel, label_writer, phrase_path, phrase, counter, phrase_counter, phrase_token_length):
    # If over, save phrase as file
    file_name = 'phrase_' + str(phrase_counter) + '.txt'
    phrase = ['<p>'] + phrase +['</p>']
    if not isdir(phrase_path):
        mkdir(phrase_path)
    with open(join(phrase_path, file_name), 'w') as f:
        f.write(phrase)
    label_writer.writerow([counter, label_novel, file_name])

    phrase = ""
    phrase_counter += 1
    counter += 1
    phrase_token_length = 0
    return phrase, phrase_token_length, phrase_counter, counter

def append_phrase(sentence, tokenizer, label_novel, label_writer, phrase_path, phrase, counter, phrase_counter, phrase_token_length):  # sentence를 phrase로 만들어주는 함수
    # nonlocal = 현재 함수의 지역 변수가 아니라는 뜻이며 바깥쪽 함수의 지역 변수를 사용
    token = tokenizer.tokenize(sentence)
    if phrase_token_length + len(token) > 1020:
        phrase, phrase_token_length, phrase_counter, counter = save_phrase(
            label_novel, label_writer, phrase_path, phrase, counter, phrase_counter, phrase_token_length)
    phrase += ['<s>'] +sentence + ['</s>'] + ' '
    phrase_token = tokenizer.tokenize(phrase)
    phrase_token_length = len(phrase_token)
    return phrase, phrase_token_length