import json
import torch
from torch.utils.data import Dataset
from torch.autograd import  Variable

# from train_model.train_NN import all_fingerprint_to_code

class NN_data_set(Dataset):
    def __init__(self,train=True):
        self.all_tensor = []
        List = []
        if(train):
            with open('train_NN_all_data_list','r') as f:
                List = json.load(f)
        else:
            with open('test_NN_all_data_list','r') as f:
                List = json.load(f)

        for item in List:
            input_code,label = item
            input_code = torch.FloatTensor(input_code)
            label = torch.FloatTensor(label)
            self.all_tensor.append([input_code,label])


        # if(train):
        #     self.all_tensor = all_fingerprint_to_code.train_NN_all_data_list
        # else:
        #     self.all_tensor = all_fingerprint_to_code.test_NN_all_data_list

    def __getitem__(self, index):
        return Variable(self.all_tensor[index][0]),Variable(self.all_tensor[index][1])

    def __len__(self):
        return len(self.all_tensor)
