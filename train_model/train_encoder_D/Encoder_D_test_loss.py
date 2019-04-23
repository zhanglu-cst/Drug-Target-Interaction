import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from train_model.train_encoder_D import Encoder_D_dataset


device = torch.device('cuda')

model = torch.load('drug_encoder_v3')

test_D_loader = DataLoader(Encoder_D_dataset.test_D_set, batch_size = 1, shuffle = True)

show_nums = 0

loss_f = nn.MSELoss()

loss_sum = 0.0
cnt = 0
for item in test_D_loader:
    batch_input = item
    batch_input = Variable(batch_input).to(device)
    code,right_output = model(batch_input)
    loss = loss_f(right_output,batch_input)
    loss_sum+=loss.data.item()
    cnt+=1

print('ave loss:{}'.format(loss_sum/cnt))


