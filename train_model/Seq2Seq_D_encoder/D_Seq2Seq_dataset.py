import json

import torch
from torch.autograd import Variable

import project_path


# 读取配置文件
def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


word_dict = ['EOS', 'SOS', 'NH', 'CH', 'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti',
             'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb',
             'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
             'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir',
             'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am',
             'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'K', 'V', 'Y',
             'I', 'W', 'U', '[', ']', '.', '#', ')', '@', '+', '=', '-', '/', '\\', '(', '>', '<pad>', '%', '1', '2',
             '3', '4', '5', '6','7','8','9','0']

def build_word2index():  # 构建有单词到下标的字典
    word2index = {}
    index = 0
    for item in word_dict:
        word2index[item] = index
        index += 1
    return word2index


word2index = build_word2index()
# with open('word2index.json','w') as f:
#     json.dump(word2index,f)
index2word = word_dict


def sentence2list(sentence):  # 将一个字符串句子转换为下标组成的list
    EOS = 0
    SOS = 1
    ans = []
    ans.append(SOS)  # 添加开始符
    i = 0
    while (i < len(sentence)):
        if (i == len(sentence) - 1):  # 如果到了最后一个，则之间查找最后一个字符的下标
            index = word2index[sentence[i]]
            ans.append(index)
            i += 1
            break
        else:
            double = sentence[i] + sentence[i + 1]  # 首先尝试查找当前字符和下一个字符组成的双字符(例如Mg,Na)是不是在字典中
            if (double in word_dict):
                ans.append(word2index[double])
                i += 2
            else:
                single = sentence[i]  # 如果当前字符和下一个字符组成的双字符不在字典中，则查找当前字符在字典中的下标
                ans.append(word2index[single])
                i += 1
    ans.append(EOS)  # 添加一个结束符
    # ans = torch.LongTensor(ans)
    return ans


def read_drug_data():
    print('Reading data ...')
    with open(project_path.train_data_raw_path, "r") as f:
        raw_lines = json.load(f)
    D_list = []
    for line in raw_lines:
        P, D, label = line
        list_D = sentence2list(D)
        D_list.append(list_D)
    return D_list


def get_minibatch(lines, word2ind, index, batch_size):
    """Prepare minibatch."""
    # 获取一个batch
    vectors = []
    max_vector_length = 0  # 记录当前batch中的最长的输入的长度，（用来将其他较短的都填充到当前batch的最长长度）

    if (index + batch_size >= len(lines)):  # 如果已经到了数据集的末尾了，则当前的batch大小更改为剩余的数据条数
        batch_size = len(lines) - index - 1

    for line in lines[index:(index + batch_size)]:  # 找到当前batch中的最长的输入的长度
        max_vector_length = max(max_vector_length, len(line))

    for line in lines[index:(index + batch_size)]:
        vector = []
        vector = vector + line

        if len(line) < max_vector_length:  # 将其他较短的都填充到当前batch的最长长度
            pad = [word2ind['<pad>']] * (max_vector_length - len(line))
            vector = vector + pad

        vectors.append(vector)

    input_lines = vectors
    label = vectors

    input_lines = Variable(torch.LongTensor(input_lines))
    label = Variable(torch.LongTensor(label))

    # 为了使用teacher forcing，decoder中的label，要与decoder的输入错开1位。
    # 这样decoder中，每一步的label就是下一步的输入。
    # 使得模型中decoder的每一步的输出都与下一步的输入计算loss，使得decoder的每一步的输出逼近下一步的输入

    label = label[:, 1:]  # 去掉开始的SOS开始符
    right_pad = [word2ind['<pad>']] * batch_size
    right_pad = Variable(torch.LongTensor(right_pad))
    right_pad = right_pad.reshape(-1, 1)
    label = torch.cat((label, right_pad), dim = 1)  # 在最右边添加一列pad

    return input_lines, label


config_file_path = './config.json'
config = read_config(config_file_path)  # 读取配置文件

D_list = read_drug_data()  # 读取数据
tmp = []
for line in D_list:
    if (len(line) < 400):
        tmp.append(line)
D_list = tmp

# cnt = 0
# for line in D_list:
#     if(len(line)>400):
#         print(len(line))
#         cnt+=1
#
# print(cnt)
# print(len(D_list))
