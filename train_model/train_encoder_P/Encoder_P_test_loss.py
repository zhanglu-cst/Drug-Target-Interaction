import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model_files import save_load_model
from train_model.train_encoder_P import Encoder_P_dataset

device = torch.device('cuda')

model = save_load_model.load('protein_encoder')
model.eval()

test_P_loader = DataLoader(Encoder_P_dataset.test_P_set, batch_size = 1, shuffle = True)

show_nums = 0

loss_f = nn.MSELoss()

loss_sum = 0.0
cnt = 0
for item in test_P_loader:
    batch_input = item
    batch_input = batch_input.to(device)
    code, right_output = model(batch_input)
    loss = loss_f(right_output, batch_input)
    loss_sum += loss.data.item()
    cnt += 1

print('ave loss:{}'.format(loss_sum / cnt))
