import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from train_model.train_encoder_D import a_Encoder_D_dataset

device = torch.device('cuda')

model = torch.load('../../model_files/drug_encoder')
test_D_loader = DataLoader(a_Encoder_D_dataset.test_D_set, batch_size = 1, shuffle = True)

show_nums = 0

for item in test_D_loader:
    batch_input = item
    batch_input = Variable(batch_input).to(device)
    code, right_output = model(batch_input)
    input_cnts = []
    output_cnts = []
    input_tensor = batch_input[0]
    print('input tensor:{}'.format(input_tensor))
    index_one_input = []
    for i, ele in enumerate(input_tensor):
        if (ele == 1):
            index_one_input.append(i)
    print('input one index:{}'.format(index_one_input))

    output_tensor = right_output[0]
    print('output tensor:{}'.format(output_tensor))
    index_one_output = []
    value_one_output = []
    for i, ele in enumerate(output_tensor):
        if (ele > 0.5):
            value_one_output.append(ele)
            index_one_output.append(i)
    print('output one index:{}'.format(index_one_output))
    print('output one value:{}'.format(value_one_output))

    input_tensor = input_tensor.reshape(-1, 1)
    output_tensor = output_tensor.reshape(-1, 1)
    stack = torch.stack((input_tensor, output_tensor), dim = 2)
    print(stack)

    go_on = input('Go on:')
    if (go_on == 'n'):
        break
