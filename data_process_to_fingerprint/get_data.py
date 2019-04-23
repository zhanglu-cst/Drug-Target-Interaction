import json

import torch
import project_path
import os

from data_process_to_fingerprint import get_fingerprint_drug
from data_process_to_fingerprint import get_fingerprint_protein
from torch.autograd import Variable
from torch.utils.data import Dataset


device = torch.device('cuda')

raw_train_path = os.path.join(project_path.PROJECT_ROOT,'data_process_to_fingerprint/train_data_raw.json')
raw_test_path = os.path.join(project_path.PROJECT_ROOT,'data_process_to_fingerprint/test_data_raw.json')

class Get_Label():
    def __init__(self):
        with open('data_process_to_fingerprint/train_data_raw.json','r') as f:
            self.train_data = json.load(f)
        with open('data_process_to_fingerprint/test_data_raw.json','r') as f:
            self.test_data = json.load(f)

    def __call__(self, P , D ):
        M = {}
        P = P[0]
        D = D[0]
        for item in self.train_data:
            if(item[0]==P and item[1]==D):
                M['label'] = item[2]
                M['set'] = '训练集'
                return M
        for item in self.test_data:
            if(item[0]==P and item[1]==D):
                M['label'] = item[2]
                M['set'] = '测试集'
                return M
        M['label'] = 'None'
        return M


def get_drug_list():
    drug_set = set()
    with open(raw_train_path,'r') as f:
        L = json.load(f)
    for item in L:
        P,D,label = item
        drug_set.add(D)
    with open(raw_test_path,'r') as f:
        L = json.load(f)
    for item in L:
        P,D,label = item
        drug_set.add(D)
    return list(drug_set)


class Drug_DataSet(Dataset):
    def __init__(self):
        all_drug_str = get_drug_list()
        self.drug_tensor = []
        self.drug_str = []
        for drug in all_drug_str:
            try:
                tmp = get_fingerprint_drug.get_fingerprint_from_smiles(drug)
                tmp = torch.FloatTensor(tmp)
                tmp = Variable(tmp).to(device)
                self.drug_tensor.append(tmp)
                self.drug_str.append(drug)
            except:
                pass


    def __getitem__(self, index):
        return self.drug_str[index],self.drug_tensor[index]

    def __len__(self):
        return len(self.drug_str)


def get_protein_list():
    protein_set = set()
    with open(raw_train_path,'r') as f:
        L = json.load(f)
    for item in L:
        P,D,label = item
        protein_set.add(P)
    with open(raw_test_path,'r') as f:
        L = json.load(f)
    for item in L:
        P,D,label = item
        protein_set.add(P)
    return list(protein_set)


class Protein_DataSet(Dataset):
    def __init__(self):
        all_protein_str = get_protein_list()
        self.protein_tensor = []
        self.protein_str = []
        for protein in all_protein_str:
            try:
                tmp = get_fingerprint_protein.get_fingerprint_from_protein_squeeze(protein)
                tmp = torch.FloatTensor(tmp)
                tmp = Variable(tmp).to(device)
                self.protein_tensor.append(tmp)
                self.protein_str.append(protein)
            except:
                pass

    def __getitem__(self, index):
        return self.protein_str[index], self.protein_tensor[index]

    def __len__(self):
        return len(self.protein_str)



def get_dict_PD_label():
    M = {}
    with open(raw_train_path,'r') as f:
        L_train = json.load(f)
    with open(raw_test_path,'r') as f:
        L_test = json.load(f)
    L = L_train+L_test
    for item in L:
        P,D,label = item
        key = P+D
        M[key]=label
    return M

class Pad_smlies_probabitity():
    def __init__(self):
        self.M = get_dict_PD_label()

    def __call__(self, smiles_probabitity, P):
        ans = []
        for item in smiles_probabitity:
            smiles = item[0]
            PD = P+smiles
            line = {}
            line['sp']=item
            if(PD in self.M):
                if(self.M[PD]==1):
                    line['real'] = True
                else:
                    line['real'] = False
            else:
                line['real']='未知'
            ans.append(line)
        return ans

class Pad_protein_probability():
    def __init__(self):
        self.M = get_dict_PD_label()

    def __call__(self, protein_probability, D):
        ans = []
        for item in protein_probability:
            P = item[0]
            PD = P+D
            line = {}
            line['pp'] = item
            if(PD in self.M):
                if(self.M[PD]==1):
                    line['real'] = True
                else:
                    line['real'] = False
            else:
                line['real'] = '未知'
            ans.append(line)
        return ans





