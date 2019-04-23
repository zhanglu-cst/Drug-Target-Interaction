import json
import os

import project_path
from data_process_to_fingerprint import get_fingerprint_drug
from data_process_to_fingerprint import get_fingerprint_protein


def transfer(dataset):
    '''return list of list [[],[],[]] of feature
    :rtype: list
    '''
    data_feature = []
    print('processing {} to fingerprints'.format(dataset))
    with open(dataset, 'r') as f:
        data_raw = json.load(f)
        for item in data_raw:
            P_str, D_str, label = item
            P_f = get_fingerprint_protein.get_fingerprint_from_protein_squeeze(P_str)
            try:
                D_f = get_fingerprint_drug.get_fingerprint_from_smiles(D_str)
                tmp1 = [P_f, D_f, label]
                data_feature.append(tmp1)
            except:
                pass
    print('fingerprints process OK!')
    return data_feature


train_fingerprint = transfer(os.path.join(project_path.PROJECT_ROOT, 'data_process_to_fingerprint/train_data_raw.json'))
test_fingerprint = transfer(os.path.join(project_path.PROJECT_ROOT, 'data_process_to_fingerprint/test_data_raw.json'))

'''train_data_feature input to encoder: ->torch.FloatTensor() -> Variable().to(device)'''
