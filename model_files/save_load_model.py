import torch
import os
from project_path import PROJECT_ROOT


def save(model,filename):
    dir_path = os.path.join(PROJECT_ROOT,'model_files')
    file_path = os.path.join(dir_path,filename)
    torch.save(model,file_path)
    print('save model {} OK'.format(filename))

def load(filename,train=False):
    dir_path = os.path.join(PROJECT_ROOT, 'model_files')
    file_path = os.path.join(dir_path, filename)
    model = torch.load(file_path)
    if(train==True):
        model.train()
    else:
        model.eval()
    return model


