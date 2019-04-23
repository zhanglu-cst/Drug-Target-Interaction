import torch
from torch.autograd import Variable
from data_process_to_fingerprint import get_fingerprint_protein
from data_process_to_fingerprint.get_data import Drug_DataSet,device
from torch.utils.data import DataLoader

from model_files import save_load_model




def sort_with_probability(lines):
    def cmp(x):
        return x[1]
    res = sorted(lines, key = cmp, reverse = True)
    return res


class search_drug():
    def __init__(self):
        self.batch_size = 1024
        self.drug_set = Drug_DataSet()
        self.drug_loader = DataLoader(self.drug_set,batch_size = self.batch_size,shuffle = False)
        self.encoder_P = save_load_model.load('protein_encoder')
        self.encoder_D = save_load_model.load('drug_encoder')
        self.NN_classify = save_load_model.load('NN_classify')

    def __call__(self, P ,threshold = 0.8):
        P = get_fingerprint_protein.get_fingerprint_from_protein_squeeze(P)
        P = [P]
        P = torch.FloatTensor(P)
        P = Variable(P).to(device)
        code_P, _ = self.encoder_P(P)

        ans = []

        for batch_items in self.drug_loader:
            batch_str,batch_tensors = batch_items
            code_D, _ = self.encoder_D(batch_tensors)

            cat_P = torch.tensor([]).to(device)
            for i in range(batch_tensors.shape[0]):
                cat_P = torch.cat((cat_P,code_P),dim = 0)

            NN_input = torch.cat((cat_P, code_D), dim = 1)
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





# s = search_drug()
# ans = s('MERYENLFAQLNDRREGAFVPFVTLGDPGIEQSLKIIDTLIDAGADALELGVPFSDPLADGPTIQNANLRAFAAGVTPAQCFEMLALIREKHPTIPIGLLMYANLVFNNGIDAFYARCEQVGVDSVLVADVPVEESAPFRQAALRHNIAPIFICPPNADDDLLRQVASYGRGYTYLLSRSGVTGAENRGALPLHHLIEKLKEYHAAPALQGFGISSPEQVSAAVRAGAAGAISGSAIVKIIEKNLASPKQMLAELRSFVSAMKAASRA')
#
# print(ans)

