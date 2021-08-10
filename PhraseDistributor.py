from os import listdir, mkdir
from os.path import isfile, isdir, join
import argparse
import csv
import kss
from tqdm import tqdm

from MrBanana_tokenizer import MyTokenizer

counter = 0

def main(dir_path: str, data_path: str, out_path: str, tokenizer):  # , data_path
    # total counter
    global counter
    # 각 paraphrase 마다 tag를 추가
    label = open(out_path + 'label.csv', 'a', newline='') # Open labeling file
    label_writer = csv.writer(label)
    label_novel = data_path.split('.')[0] # 소설 제목
    # Open original novel file and Load text
    novel = open(dir_path + data_path, 'r', encoding='utf-8')
    text = novel.read()
    text = text.replace("\n", "") # 인소에 있는 특정 단어들 빼주기
    text = text.replace("Aa", "")
    novel.close()
    
    split_list = kss.split_sentences(text) # Split sentences

    phrase_path = out_path + label_novel
    phrase_counter = 0
    phrase = ""
    phrase_token_length = 0

    def append_phrase(sentence, tokenizer): # sentence를 phrase로 만들어주는 함수
        nonlocal phrase_token_length # nonlocal = 현재 함수의 지역 변수가 아니라는 뜻이며 바깥쪽 함수의 지역 변수를 사용
        nonlocal phrase
        token = tokenizer.tokenize(sentence)
        if phrase_token_length + len(token) > 1022:
            save_phrase()
        phrase += sentence + ' '
        phrase_token = tokenizer.tokenize(phrase)
        phrase_token_length = len(phrase_token)

    def save_phrase():
        global counter
        nonlocal label_novel # 소설 제목
        nonlocal label_writer # csv writer
        nonlocal phrase_path
        nonlocal phrase
        nonlocal phrase_counter # phrase tag 번호
        nonlocal phrase_token_length
        # If over, save phrase as file
        file_name = 'phrase_' + str(phrase_counter) + '.txt'
        if not isdir(phrase_path):
            mkdir(phrase_path)
        with open(join(phrase_path, file_name), 'w') as f:
            f.write(phrase)
        label_writer.writerow([counter, label_novel, file_name])
        
        phrase = ""
        phrase_counter += 1
        counter += 1
        phrase_token_length = 0

    for line in tqdm(split_list): # sentence list 안에서
        # Check phrase size
        tok = tokenizer.tokenize(line)
        # Add sentence to phrase
        if len(tok) < 1022:
            append_phrase(line, tokenizer)
        else: # 더 길면
            sent = ''
            flag = False # 대사의 큰, 작은 따옴표를 한 분장으로 처리하기 위해
            err = False
            for e in line:
                sent += e
                if not flag and (e == '"' or e == '“'):
                    flag = True
                elif flag and (e == '"' or e == '”' or e == '’' or e == "'"):
                    flag = False
                    sent = sent.strip(' ') # strip() : 문자열에서 특정 문자 제거
                    sent = sent.strip('\n')
                    append_phrase(sent, tokenizer)
                    sent = '' # 초기화
                elif not flag and e == '.':
                    sent = sent.strip(' ')
                    sent = sent.strip('\n')
                    append_phrase(sent, tokenizer)
                    sent = ''
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
                if len(tokenizer.tokenize(sent)) > 1022:
                    sent = ''
                    break
            if sent:
                sent = sent.strip(' ')
                sent = sent.strip('\n')
                append_phrase(sent, tokenizer)
                sent = ''

    if phrase:
        save_phrase()
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

    # Setup tokenizer
    vocab_file_path = './tokenizer/vocab.json'
    merge_file_path = './tokenizer/merges.txt'

    tokenizer = MyTokenizer(vocab_file_path, merge_file_path)

    # List of novel files
    novel_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    # Distribute Phrase
    for novel in novel_files:
        main(dir_path, novel, out_path, tokenizer)
