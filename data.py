# Ref: Narrative GPT-2
import os
from tqdm import tqdm
import kss
from MrBanana_tokenizer import MyTokenizer

vocab_file_path = './tokenizer/vocab.json'
merge_file_path = './tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)

# kss sentence -> txt splitter (MY)
def load_data(data_path, out_dir):  # , data_path
  file = open(data_path, 'r', encoding='utf-8')
  text = file.read()
  file.close()
  text = text.replace("\n", "")
  text = text.replace("Aa", "")
  
  split_list = kss.split_sentences(text)
  i = 56537
  for line in tqdm(split_list):
    tok = tokenizer.tokenize(line)
    if len(tok) < 1022:
      file_name = out_dir + '/novel_sentence' + str(i) + '.txt'
      output = open(file_name, 'w')
      line = line.strip(' ')
      line = line.strip('\n')
      output.write(line)
      i += 1
    else:
      sent = ''
      flag = False
      err = False
      for e in line:
        sent += e
        if not flag and (e == '"' or e == '“'):
          flag = True
        elif flag and (e == '"' or e == '”' or e == '’' or e == "'"):
          flag = False
          file_name = out_dir + '/novel_sentence' + str(i) + '.txt'
          output = open(file_name, 'w')
          sent = sent.strip(' ')
          sent = sent.strip('\n')
          output.write(sent)
          sent = ''
          i += 1
        elif not flag and e == '.':
          file_name = out_dir + '/novel_sentence' + str(i) + '.txt'
          output = open(file_name, 'w')
          sent = sent.strip(' ')
          sent = sent.strip('\n')
          output.write(sent)
          sent = ''
          i += 1
        elif flag and len(sent) == 1000:
          flag = False
        # elif len(sent) == 1022:
        #   sent = sent.strip(' ')
        #   sent = sent.strip('\n')
        #   output.write(sent)
        #   print("rare1")
        else:
          err=True
          assert err == True
    output.close()

def load_total(input_file_path, output_path):
  data_list = os.listdir(input_file_path)
  for path in data_list:
    data_path = input_file_path + '/' + path
    load_data(data_path, output_path)

# def sentence_splitter(file_name):
#   file = open(file_name, 'r', encoding='utf-8')
#   splitted_file = open('GPT2_dataset/splitted_test1.txt', 'w', encoding='utf-8')
#   text = file.read()
#   text = text.replace("\n","")
#
#   split_list = kss.split_sentences(text)
#   data = []
#   for l in split_list:
#     if len(l) < 1024:
#       # padded_data = tokenized_line + ['<pad>'] * (1024 - len(tokenized_line))
#       data.append(l)
#     else:
#       sent = ""
#       flag = False
#       for e in l:
#         sent += e
#         if not flag and (e == '"' or e == '“'):
#           flag = True
#         elif flag and (e == '"' or e == '”' or e == '’'or e == "'"):
#           flag = False
#           data.append(sent)
#           sent = ""
#         elif not flag and e == '.':
#           data.append(sent)
#           sent = ""
#         elif flag and len(sent) == 1000:
#           flag = True
#
#   for line in data:
#     splitted_file.write(line+'\n')
#
#   file.close()
#   splitted_file.close()

# if __name__ == "__main__":
  # input_dir = 'roon'
  # output_dir = 'GPT2_dataset_roon'
  # load_total(input_dir, output_dir)
  
  # merge_file_path의 길이
  # with open(merge_file_path, 'r') as file:
  #   print(len(file.readlines())) # 50996-1
  
  # vocab_file_path의 길이
  # import json
  # with open(vocab_file_path, 'r') as f:
  #   data = json.loads(f.read())
  #   print(len(data)) # 52000