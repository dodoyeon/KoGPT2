from os import listdir, mkdir
from os.path import isfile, isdir, join

import argparse

import csv
import kss
from numpy import empty
from tqdm import tqdm
from MrBanana_tokenizer import MyTokenizer

vocab_file_path = './tokenizer/vocab.json'
merge_file_path = './tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
counter = 0

def main(dir_path: str, data_path: str, out_path: str):  # , data_path
    # total counter
    global counter
    # Open labeling file 
    label = open(out_path + 'label.csv', 'w', newline='')
    label_writer = csv.writer(label)
    label_novel = data_path.split('.')[0]
    # Open original novel file and Load text
    novel = open(dir_path + data_path, 'r', encoding='utf-8')
    text = novel.read()
    text = text.replace("\n", "")
    text = text.replace("Aa", "")
    novel.close()
    # Split sentences
    split_list = kss.split_sentences(text)

    phrase_path = out_path + label_novel
    phrase_counter = 0
    phrase = ""
    for line in tqdm(split_list):
        # Check phrase size
        tok = tokenizer.tokenize(line)
        if len(phrase) + len(tok) >= 1022:
            # If over, save phrase as file
            file_name = 'phrase_' + str(phrase_counter) + '.txt'
            if not isdir(phrase_path):
                mkdir(phrase_path)
            with open(phrase_path + '/' + file_name, 'w') as f:
                f.write(phrase)
                label_writer.writerow([counter, label_novel, file_name])
            phrase = ""
            phrase_counter += 1
            counter = counter + 1
        else:
            # If not, add sentence to phrase
            sent = ''
            flag = False
            err = False
            for e in line:
                sent += e
                if not flag and (e == '"' or e == '“'):
                    flag = True
                elif flag and (e == '"' or e == '”' or e == '’' or e == "'"):
                    flag = False
                    sent = sent.strip(' ')
                    sent = sent.strip('\n')
                    phrase += sent + ' '
                elif not flag and e == '.':
                    sent = sent.strip(' ')
                    sent = sent.strip('\n')
                    phrase += sent + ' '
                elif flag and len(sent) == 1000:
                    flag = False
                # elif len(sent) == 1022:
                #   sent = sent.strip(' ')
                #   sent = sent.strip('\n')
                #   output.write(sent)
                #   print("rare1")
                else:
                    err = True
                    assert err == True
    if phrase:
        file_name = 'phrase_' + str(phrase_counter) + '.txt'
        if not isdir(phrase_path):
            mkdir(phrase_path)
        with open(phrase_path + '/' + file_name, 'w') as f:
            f.write(phrase)
            label_writer.writerow([counter, label_novel, file_name])
    label.close()

if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str, default='GPT2_dataset_original/',
                        help='Original novel directory', dest='dir_path')
    parser.add_argument('--output', '-o', type=str, default='phraseDataset/',
                        help='Output directory', dest='out_path')
    args= parser.parse_args()

    # Assign arguments to variables
    dir_path = args.dir_path
    out_path = args.out_path

    # List of novel files
    novel_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    # Distribute Phrase
    for novel in novel_files:
        main(dir_path, novel, out_path)
