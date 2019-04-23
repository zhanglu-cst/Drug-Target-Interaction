import json

import torch
from torch.autograd import Variable

from data_process_to_fingerprint import  ALL_rawstr_to_fingerprint
from model_files import save_load_model

device = torch.device('cuda')

model_Encoder_D = save_load_model.load('drug_encoder')
model_Encoder_P = save_load_model.load('protein_encoder')


def transfer_to_tensor(x):
    tensor = torch.FloatTensor(x)
    tensor = tensor.unsqueeze(0)
    tensor = Variable(tensor).to(device)
    return tensor


def get_input_of_NN(dataset_feature):
    print('processing...')
    all_data = []
    cnt = 0
    for item in dataset_feature:
        P_f, D_f, label = item
        P_f = transfer_to_tensor(P_f)
        P_code, right_out = model_Encoder_P(P_f)
        D_f = transfer_to_tensor(D_f)
        D_code, right_out = model_Encoder_D(D_f)
        P_code = P_code.squeeze(0)
        D_code = D_code.squeeze(0)
        code_input = torch.cat((P_code, D_code))
        code_input = code_input.tolist()
        label = [label]
        # label = torch.FloatTensor(label)
        all_data.append([code_input, label])

        cnt += 1
        if (cnt % 1000 == 0):
            print('cnt:{},total:{}'.format(cnt, len(dataset_feature)))

    return all_data


train_NN_all_data_list = get_input_of_NN(ALL_rawstr_to_fingerprint.train_fingerprint)
test_NN_all_data_list = get_input_of_NN(ALL_rawstr_to_fingerprint.test_fingerprint)

with open('train_NN_all_data_list','w') as f:
    json.dump(train_NN_all_data_list,f)

with open('test_NN_all_data_list','w') as f:
    json.dump(test_NN_all_data_list,f)

print('save ok')


