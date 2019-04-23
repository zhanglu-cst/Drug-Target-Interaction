import json

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from train_model.train_NN.NN_data_set import NN_data_set
from model_files import save_load_model

device = torch.device('cuda')

test_set = NN_data_set(train = False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = True)

model = save_load_model.load('NN_classify')
model = model.to(device)
model.eval()

sum = 0
correct = 0
all_out = []
for item in test_loader:
    batch_inputs, batch_labels = item
    batch_inputs = Variable(batch_inputs).to(device)
    batch_labels = Variable(batch_labels).to(device)

    out = model(batch_inputs)
    yuzhi = 0
    cur_label = batch_labels[0].data.item()
    cur_out = out[0].data.item()
    all_out.append([cur_out,cur_label])
    # if ((cur_out > yuzhi and cur_label == 1) or (cur_out < yuzhi and cur_label == 0)):
    #     correct += 1
    print(cur_out,cur_label)
    sum += 1

print(all_out)
with open('all_out.json','w') as f:
    json.dump(all_out,f)


print('acc:{}'.format(correct / sum))
