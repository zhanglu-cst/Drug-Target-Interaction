import torch
from torch.autograd import Variable
from data_process_to_fingerprint import get_fingerprint_drug
from data_process_to_fingerprint.get_data import device,Protein_DataSet
from torch.utils.data import DataLoader

from model_files import save_load_model



def sort_with_probability(lines):
    def cmp(x):
        return x[1]
    res = sorted(lines, key = cmp, reverse = True)
    return res


class search_protein():
    def __init__(self):
        self.batch_size = 1024
        self.protein_set = Protein_DataSet()
        self.protein_loader = DataLoader(self.protein_set,batch_size = self.batch_size,shuffle = False)
        self.encoder_P = save_load_model.load('protein_encoder')
        self.encoder_D = save_load_model.load('drug_encoder')
        self.NN_classify = save_load_model.load('NN_classify')

    def __call__(self, D ,threshold = 0.8):
        try:
            D = get_fingerprint_drug.get_fingerprint_from_smiles(D)
        except:
            raise Exception('输入的药物SMILES表达式不合法')
        D = [D]
        D = torch.FloatTensor(D)
        D = Variable(D).to(device)
        code_D,_ = self.encoder_D(D)

        ans = []

        for batch_items in self.protein_loader:
            batch_str,batch_tensors = batch_items
            code_P, _ = self.encoder_P(batch_tensors)

            cat_D = torch.tensor([]).to(device)
            for i in range(batch_tensors.shape[0]):
                cat_D = torch.cat((cat_D,code_D),dim = 0)

            NN_input = torch.cat((code_P,cat_D), dim = 1)
            out = self.NN_classify(NN_input)
            out = out.squeeze(1)
            out = out.data.tolist()
            for i,probability in enumerate(out):
                if(probability>threshold):
                    string = batch_str[i]
                    res= [string,probability]
                    ans.append(res)
        ans = sort_with_probability(ans)
        return ans



# s = search_protein()
# ans = s('NNC(=O)C1=CC=NC=C1')
# print(ans)