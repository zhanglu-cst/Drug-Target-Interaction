import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda')

def get_DataLoader(lines_list, word2index, batch_size = 2):
    class Seq2Seq_Dataset(Dataset):
        def __init__(self, lines_list):
            self.lines = lines_list

        def __getitem__(self, index):
            return self.lines[index]

        def __len__(self):
            return len(self.lines)

    def collate_fn(batch):
        max_vector_length = 0
        for line in batch:
            max_vector_length = max(max_vector_length, len(line))
        print('max len:{}'.format(max_vector_length))

        new_batch = []
        for line in batch:
            tmp_line = line
            if (len(line) < max_vector_length):
                pad = [word2index['<pad>']] * (max_vector_length - len(line))
                tmp_line = tmp_line + pad
            new_batch.append(tmp_line)

        input_lines = Variable(torch.LongTensor(new_batch))
        label = Variable(torch.LongTensor(new_batch))
        label = label[:, 1:]  # 去掉开始的SOS开始符
        right_pad = [word2index['<pad>']] * input_lines.shape[0]
        right_pad = Variable(torch.LongTensor(right_pad))
        right_pad = right_pad.reshape(-1, 1)
        label = torch.cat((label, right_pad), dim = 1)


        input_lines = input_lines.to(device)
        label = label.to(device)
        return input_lines,label



    dataset = Seq2Seq_Dataset(lines_list)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)

    return loader


