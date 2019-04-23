import json

import project_path

word_dict = ['EOS', 'SOS', 'M', 'E', 'R', 'Y', 'N', 'L', 'F', 'A', 'Q', 'D', 'G', 'V', 'P', 'T', 'I', 'S', 'K', 'C',
             'H', 'W', 'U', 'X', '<pad>']


def build_word2index():  # 构建有单词到下标的字典
    word2index = {}
    index = 0
    for item in word_dict:
        word2index[item] = index
        index += 1
    return word2index


word2index = build_word2index()
index2word = word_dict


def read_drug_data():
    print('Reading data ...')
    with open(project_path.train_data_raw_path, "r") as f:
        raw_lines = json.load(f)
    P_raw_list = []
    for line in raw_lines:
        P, D, label = line
        P_raw_list.append(P)
    return P_raw_list


P_raw_list = read_drug_data()


def get_list_P():
    list_P = []
    EOS = word2index['EOS']
    SOS = word2index['SOS']
    for line in P_raw_list:
        ans = []
        ans.append(SOS)
        for ch in line:
            ans.append(word2index[ch])
        ans.append(EOS)
        list_P.append(ans)
    return list_P


list_P = get_list_P()

del P_raw_list

# 读取配置文件
def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


config_file_path = './config.json'
config = read_config(config_file_path)  # 读取配置文件
