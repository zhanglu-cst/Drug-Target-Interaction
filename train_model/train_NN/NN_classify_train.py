import json
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import sys
sys.path.append('..')

from train_model.train_NN.NN_data_set import NN_data_set
from train_model.train_NN.NN_classify_Net import NN
# from model_files import save_load_model

device = torch.device('cuda')


train_set = NN_data_set(train = True)
train_loader = DataLoader(train_set,batch_size = 1024,shuffle = True)


model = NN()
# model = save_load_model.load('NN_classify', train = True)
print(model)
model.train()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

test_set = NN_data_set(train = False)
test_loader = DataLoader(test_set,batch_size = 1)

def Get_ACC():
    cnt = 0
    correct = 0
    model.eval()
    loss_sum = 0
    for item in test_loader:
        batch_inputs, batch_labels = item
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        out = model(batch_inputs)
        loss_sum+=criterion(out,batch_labels)
        yu_zhi = 0.5
        cur_label = batch_labels[0].data.item()
        cur_out = out[0].data.item()
        if ((cur_out > yu_zhi and cur_label == 1) or (cur_out < yu_zhi and cur_label == 0)):
            correct += 1
        cnt += 1
    model.train()
    acc = correct / cnt *100
    eval_loss = loss_sum/cnt
    return acc,eval_loss



best_model = {'acc':0,'loss':0}

py = []

# def plot(print_loss):
#     plt.figure(1)
#     plt.cla()
#     py.append(print_loss)
#     plt.plot(py, 'go-', linewidth=2, markersize=4,
#              markeredgecolor='red', markerfacecolor='m')
#     plt.pause(0.000000001)

py_acc = []
# def plot_acc(acc):
#     plt.figure(2)
#     plt.cla()
#     py_acc.append(acc)
#     plt.plot(py_acc, 'go-', linewidth = 2, markersize = 4,
#              markeredgecolor = 'red', markerfacecolor = 'm')
#     plt.pause(0.000000001)


# acc,eval_loss = Get_ACC()
# # plot_acc(acc)
# py_acc.append(acc)

for epoch in range(5000):
    cnt = 0
    sum_loss = 0
    for item in train_loader:
        batch_inputs,batch_labels = item
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        out = model(batch_inputs)
        loss = criterion(out,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
        cnt+=1
    ave_loss = sum_loss/cnt
    print('epoch:{},loss:{}'.format(epoch, ave_loss))

    if((epoch+1)%50==0):
        py.append(ave_loss)
        acc,eval_loss = Get_ACC()
        print('eval: acc:{}, eval loss:{}'.format(acc,eval_loss))
        print('best acc:{}, -> eval loss:{}'.format(best_model['acc'],best_model['loss']))
        py_acc.append(acc)
        if(acc>best_model['acc']):
            best_model = {'acc':acc, 'loss':eval_loss}
            print('this is best!')
            # save_load_model.save(model, 'NN_classify')


print('best acc:{}, best eval loss:{}'.format(best_model['acc'],best_model['loss']))


with open('py.json','w') as f:
    json.dump(py,f)
with open('py_acc.json','w') as f:
    json.dump(py_acc,f)


# plt.figure(1)
# plt.cla()
# plt.plot(py, 'go-', linewidth=2, markersize=4,
#          markeredgecolor='red', markerfacecolor='m')
# plt.pause(0.000000001)
#
# plt.figure(2)
# plt.cla()
# plt.plot(py_acc, 'go-', linewidth = 2, markersize = 4,
#          markeredgecolor = 'red', markerfacecolor = 'm')
# plt.pause(0.000000001)
#
# plt.show()


