import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from data_process_to_fingerprint import ALL_rawstr_to_fingerprint


class MyDataSet(Dataset):
    def __init__(self, train = True):
        all_P_input = []
        if (train):
            for item in ALL_rawstr_to_fingerprint.train_fingerprint:
                P_f, D_f, label = item
                all_P_input.append(P_f)
        else:
            for item in ALL_rawstr_to_fingerprint.test_fingerprint:
                P_f, D_f, label = item
                all_P_input.append(P_f)
        print('list Protein len:{}'.format(len(all_P_input)))
        self.P_tensor = torch.FloatTensor(all_P_input)

    def __getitem__(self, index):
        return Variable(self.P_tensor[index])

    def __len__(self):
        return len(self.P_tensor)


train_P_set = MyDataSet(train = True)
test_P_set = MyDataSet(train = False)

