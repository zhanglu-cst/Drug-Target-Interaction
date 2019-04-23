import torch.optim as optim
import time

from model_files import save_load_model
from train_model.Seq2Seq_P_encoder import P_get_dataloader
from train_model.Seq2Seq_P_encoder.P_Seq2Seq_dataset import *
from train_model.Seq2Seq_P_encoder.P_Seq2Seq_model import *

device = torch.device('cuda')  # 选择使用的设备(cuda:使用GPU, cpu:使用cpu)

vocab_size = len(word2index)  # 单词的数量（词典大小）

weight_mask = torch.ones(vocab_size).to(device)
weight_mask[word2index['<pad>']] = 0  # loss的权重设置，这一句和上一句的作用是，将填充的pad的权重设置为0，其他的权重设置为1
loss_criterion = nn.CrossEntropyLoss(weight = weight_mask).to(device)  # 定义loss，多分类损失。CrossEntropyLoss内部带有softmax，所以模型不需要做softmax

try:
    model = save_load_model.load('Seq2Seq_P_encoder')
    model.train()
    print('Load Model OK!')
except:
    model = Seq2SeqAttention(
            src_emb_dim = config['model']['dim_word_src'],
            trg_emb_dim = config['model']['dim_word_trg'],
            src_vocab_size = vocab_size,
            trg_vocab_size = vocab_size,
            src_hidden_dim = config['model']['dim'],
            trg_hidden_dim = config['model']['dim'],
            ctx_hidden_dim = config['model']['dim'],
            attention_mode = 'dot',
            bidirectional = config['model']['bidirectional'],
            pad_token_src = word2index['<pad>'],
            pad_token_trg = word2index['<pad>'],
            nlayers = config['model']['n_layers_src'],
            nlayers_trg = config['model']['n_layers_trg'],
            dropout = 0.,
    ).to(device)  # 如果载入失败，就重新新建一个模型
    print('New Model!')

optimizer = optim.Adam(model.parameters())  # 定义优化器

dataloader = P_get_dataloader.get_DataLoader(list_P, word2index)

best_loss = 100
for epoch in range(0, 10):  # epoch循环
    cnt = 0
    for item in dataloader:
        time_start = time.time()
        batch_input_lines, batch_label = item


        decoder_logit = model(batch_input_lines, batch_input_lines)  # decoder_logit为模型的预测输出

        loss = loss_criterion(
                decoder_logit.contiguous().view(-1, vocab_size),
                batch_label.view(-1)
        )

        loss.backward()

        if ((cnt + 1) % 1 == 0):
            optimizer.step()
            optimizer.zero_grad()

        if ((cnt+1) % 10 == 0):
            word_probs = model.decode(decoder_logit).data.cpu().numpy().argmax(axis = -1)
            real = batch_label.data.cpu().numpy()
            print('teacher forcing result')
            for sentence_pred, sentence_real in zip(word_probs[:1], real[:1]):
                sentence_pred = [index2word[x] for x in sentence_pred]
                sentence_real = [index2word[x] for x in sentence_real]
                print('label:\n{}'.format(' '.join(sentence_real)))
                print('predict:\n{}'.format(' '.join(sentence_pred)))

        if ((cnt + 1) % 10 == 0):
            save_load_model.save(model, 'Seq2Seq_P_encoder')
            print('save model OK..........!!!!!')

        torch.cuda.empty_cache()

        cnt += 1
        time_end = time.time()
        print('epoch:{},cnt:{},time:{:.2f},loss:{}'.format(epoch, cnt,time_end-time_start, loss.data.item()))

torch.save(model, 'model')
print('save model OK!')
