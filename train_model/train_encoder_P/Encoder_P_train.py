import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from train_model.train_encoder_P import Encoder_P_dataset
from train_model.train_encoder_P.Encoder_P_Net import AutoEncoder
from model_files import save_load_model


device = torch.device('cuda')

train_P_loader = DataLoader(Encoder_P_dataset.train_P_set, batch_size = 1024, shuffle = True)

model = AutoEncoder()
model = model.to(device)
model.train()
print(model)

loss_f = nn.MSELoss()
optimizer = optim.Adam(model.parameters())


def Get_ACC():
    test_P_loader = DataLoader(Encoder_P_dataset.test_P_set, batch_size = 1024, shuffle = True)
    loss_sum = 0.0
    cnt = 0
    model.eval()
    for item in test_P_loader:
        batch_input = item
        batch_input = batch_input.to(device)
        code, right_output = model(batch_input)
        loss = loss_f(right_output, batch_input)
        loss_sum += loss.data.item()
        cnt += 1
    model.train()
    return (loss_sum / cnt)

best_model = [0]
best_eval_loss = 100

for epoch in range(1000):
    cnt = 0
    ave_loss = 0
    for item in train_P_loader:
        batch_input = item
        batch_input = batch_input.to(device)
        code,right_output = model(batch_input)
        loss = loss_f(right_output,batch_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        # print('print_loss:{}.'.format(print_loss))
        ave_loss+=print_loss
        cnt+=1
    print('epoch:{},loss:{}'.format(epoch, ave_loss/cnt))
    if(epoch%10==0):
        eval_loss = Get_ACC()
        print('eval loss:{}'.format(eval_loss))
        if(eval_loss<best_eval_loss):
            best_model.pop()
            best_model.append(model)
            best_eval_loss = eval_loss
            print('best loss:{}'.format(best_eval_loss))


model = best_model[0]
print('best eval loss:{}'.format(best_eval_loss))
save_load_model.save(model, 'protein_encoder')