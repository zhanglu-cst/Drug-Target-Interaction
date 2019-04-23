import torch
from torch.autograd import Variable
from data_process_to_fingerprint import get_fingerprint_drug
from data_process_to_fingerprint import get_fingerprint_protein
from model_files import save_load_model


device = torch.device('cuda')


class Model_API():
    def __init__(self):
        self.encoder_P = save_load_model.load('protein_encoder')
        self.encoder_D = save_load_model.load('drug_encoder')
        self.NN_classify = save_load_model.load('NN_classify')


    def __call__(self,list_P,list_D,sequence=True):
        if(sequence==False):
            pass
            # TODO transfer P,D id in list to seq
        fingerprint_P,fingerprint_D = [],[]
        if(len(list_P)!=len(list_D)):
            print('length of protein and drug are not equal')
            return

        #transfer to fingerprint
        for i in range(0,len(list_D)):
            P_str = list_P[i]
            D_str = list_D[i]
            fingerprint_P.append(get_fingerprint_protein.get_fingerprint_from_protein_squeeze(P_str))
            try:
                fingerprint_D.append(get_fingerprint_drug.get_fingerprint_from_smiles(D_str))
            except:
                raise Exception('药物SMILES表达式不合法'.format(D_str))
        tensor_P = torch.FloatTensor(fingerprint_P)
        tensor_D = torch.FloatTensor(fingerprint_D)
        tensor_P = Variable(tensor_P).to(device)
        tensor_D = Variable(tensor_D).to(device)

        # encode
        code_P,right_out = self.encoder_P(tensor_P)
        code_D,right_out = self.encoder_D(tensor_D)


        #cat classify
        NN_input = torch.cat((code_P,code_D),dim = 1)
        out = self.NN_classify(NN_input)
        out = out.squeeze(1)
        value = out.data.tolist()
        return value


# api = Model_API()

# v = api(['ABDWA','AAAA'],['CC','NO'])
# print(v)

