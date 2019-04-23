import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from train_model.Seq2Seq_D_encoder.D_Seq2Seq_dataset import config

device = torch.device('cuda')


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias = False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias = False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first = True):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask = None):
        """Propogate input through the network."""

        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                    self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))  # 列循环
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

    def decode_batch_step(self, input, hidden, ctx, ctx_mask = None):
        # 在对测试集进行预测的过程中使用该函数
        # 该函数传入一列输入，和隐含层，对其只执行1步的解码。返回当前解码结果和隐含层

        def recurrence(input, hidden):
            # 进行解码
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                    self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:  #如果batch在第0维，则将batch转换到第1维，将seq len转换到第0维
            input = input.transpose(0, 1)


        output = []

        right_input = input[0]  #该函数只对一列进行解码，因此input的是一列，取出这一列，相当于去掉shape中的第0维，也可以用squeeze(0)代替
        hidden = recurrence(right_input, hidden)

        output.append(hidden[0])

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:    #同样，如果batch在第0维,转换回去
            output = output.transpose(0, 1)

        return output, hidden


class Seq2SeqAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
            self,
            src_emb_dim,
            trg_emb_dim,
            src_vocab_size,
            trg_vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            ctx_hidden_dim,
            attention_mode,
            pad_token_src,
            pad_token_trg,
            bidirectional = True,
            nlayers = 2,
            nlayers_trg = 2,
            dropout = 0.,
    ):
        """Initialize model."""
        super(Seq2SeqAttention, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.ctx_hidden_dim = ctx_hidden_dim
        self.attention_mode = attention_mode
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg

        self.src_embedding = nn.Embedding(
                src_vocab_size,
                src_emb_dim,
                self.pad_token_src
        )
        self.trg_embedding = nn.Embedding(
                trg_vocab_size,
                trg_emb_dim,
                self.pad_token_trg
        )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim
        self.encoder = nn.LSTM(
                src_emb_dim,
                self.src_hidden_dim,
                nlayers,
                bidirectional = bidirectional,
                batch_first = True,
                dropout = self.dropout
        )

        self.decoder = LSTMAttentionDot(
                trg_emb_dim,
                trg_hidden_dim,
                batch_first = True
        )

        self.encoder2decoder = nn.Linear(
                self.src_hidden_dim * self.num_directions,
                trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
                self.encoder.num_layers * self.num_directions,
                batch_size,
                self.src_hidden_dim
        ), requires_grad = False).to(device)
        c0_encoder = Variable(torch.zeros(
                self.encoder.num_layers * self.num_directions,
                batch_size,
                self.src_hidden_dim
        ), requires_grad = False).to(device)

        return h0_encoder, c0_encoder

    def forward(self, input_src, input_trg, trg_mask = None, ctx_mask = None):
        """Propogate input through the network."""
        # input_src:当前的encoder的输入  ，input_src.shape  :[batch_size,max_seq_len]
        # input_trg:当前的decoder的输入，（使用teacher forcing）

        src_emb = self.src_embedding(input_src)  # 词嵌入
        trg_emb = self.trg_embedding(input_trg)

        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        # 进行编码
        src_h, (src_h_t, src_c_t) = self.encoder(
                src_emb, (self.h0_encoder, self.c0_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        # print('decoder_init_state shape:{}'.format(decoder_init_state.shape))  # [batch,1000]

        ctx = src_h.transpose(0, 1)

        # trg_h: decoder的所有输出，hnn,cnn: decoder最后的隐含层和状态层
        trg_h, (hnn, cnn) = self.decoder(
                trg_emb,
                (decoder_init_state, c_t),
                ctx,
                ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
                trg_h.size()[0] * trg_h.size()[1],
                trg_h.size()[2]
        )

        # 将deocder的输出用回归映射到单词空间
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
                trg_h.size()[0],
                trg_h.size()[1],
                decoder_logit.size()[1]
        )
        return decoder_logit


    def decode_batch_input(self, input_src, init_input_trg, ctx_mask = None):
        #该函数在训练的时候用不到，在对测试集进行预测的时候使用。
        # input_src：为encdoer的输入序列（即化学反应的产物+试剂）
        # init_input_trg: 初始的decoder输入，即SOS，由于对batch解码，因此shape为[batch_size,1]

        src_emb = self.src_embedding(input_src)  #对输入序列进行词嵌入
        right_input_emb = self.trg_embedding(init_input_trg)  # 对初始的decoder输入进行词嵌入

        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        #对输入序列进行编码
        src_h, (src_h_t, src_c_t) = self.encoder(
                src_emb, (self.h0_encoder, self.c0_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))


        # 由于batch为第0个维度，因此将第0个维度和第1个维度交换
        ctx = src_h.transpose(0, 1)
        hidden = (decoder_init_state, c_t)
        steps = config['data']['max_trg_length']   #定义一个decoder的最长序列长度，一旦达到了这个长度，停止解码

        ans = torch.LongTensor([]).to(device)   #保存结果

        batch_OK = numpy.zeros(init_input_trg.shape[0])  #标记batch中，哪些行已经出现了EOS结束符，如果所有的行都出现了EOS，则提前结束解码
        for i in range(steps):

            # decode_batch_step为对当前的一列进行解码
            trg_h, hidden = self.decoder.decode_batch_step(
                    right_input_emb,
                    hidden,
                    ctx,
                    ctx_mask
            )
            #将输出reshape
            trg_h_reshape = trg_h.contiguous().view(
                    trg_h.size()[0] * trg_h.size()[1],
                    trg_h.size()[2]
            )
            # 将输出linear到单词空间
            decoder_logit = self.decoder2vocab(trg_h_reshape)
            decoder_logit = decoder_logit.view(
                    trg_h.size()[0],
                    trg_h.size()[1],
                    decoder_logit.size()[1]
            )
            word_probs = self.decode(decoder_logit)  #执行一个softmax，转换为概率
            # 找到概率最大的单词，作为下一个单词
            decoder_argmax = word_probs.data.cpu().numpy().argmax(axis = -1)
            next_preds = Variable(
                    torch.from_numpy(decoder_argmax[:, -1])
            ).to(device)
            next_preds = next_preds.reshape(-1, 1)
            #重新进行词嵌入，继续循环，继续decode
            right_input_emb = self.trg_embedding(next_preds)

            # 将当前的结果添加当ans中
            ans = torch.cat((ans, next_preds), dim = 1)

            # 看一下，当前batch中是不是所有的行都出现了EOS
            for k in range(init_input_trg.shape[0]):
                tmp = next_preds[k].item()
                if (tmp == 0):  # EOS->0
                    batch_OK[k] = 1  # marked with 1
            if (numpy.sum(batch_OK) == init_input_trg.shape[0]):  # 所有的行都出现了EOS
                break

        return ans


    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
                logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs
