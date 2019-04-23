import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from train_model.train_encoder_D import Encoder_D_dataset
from train_model.train_encoder_D.Encoder_D_Net import AutoEncoder
from model_files import save_load_model

device = torch.device('cuda')

model = AutoEncoder()
model = model.to(device)
model.train()

train_D_loader = DataLoader(Encoder_D_dataset.train_D_set, batch_size = 2048, shuffle = True)

loss_f = nn.BCELoss()
optimizer = optim.Adam(model.parameters())


def Get_ACC():
    test_D_loader = DataLoader(Encoder_D_dataset.test_D_set, batch_size = 1024, shuffle = True)
    loss_f = nn.MSELoss()
    loss_sum = 0.0
    cnt = 0
    for item in test_D_loader:
        batch_input = item
        batch_input = batch_input.to(device)
        code, right_output = model(batch_input)
        loss = loss_f(right_output, batch_input)
        loss_sum += loss.data.item()
        cnt += 1
    return (loss_sum / cnt)


for epoch in range(500):
    cnt = 0
    sum_loss = 0
    for item in train_D_loader:
        batch_input = item
        batch_input = batch_input.to(device)
        code, right_output = model(batch_input)
        loss = loss_f(right_output, batch_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        cnt += 1
    train_loss = sum_loss / cnt
    print('epoch:{},train loss:{}'.format(epoch, train_loss))
    if(epoch%10==0):
        eval_loss = Get_ACC()
        print('eval loss:{}'.format(eval_loss))
        if(train_loss<0.0008 and eval_loss<0.0008):
            break


save_load_model.save(model, 'drug_encoder')


