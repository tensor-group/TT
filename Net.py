import torch
import logging
import numpy as np
import torchvision
import torch.nn as TorchNN
import torch.autograd as TorchAutograd
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.functional as TorchNNFun
from modules.ttrnn import StatefulTTLSTMCell, StatefulTTGRUCell

class LSTM(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, num_class, bias=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        #self.w_f_size = (feature_size + hidden_size, hidden_size)
        #self.w_i_size = (feature_size + hidden_size, hidden_size)
        #self.w_c1_size = (feature_size + hidden_size, hidden_size)
        #self.w_o_size = (feature_size + hidden_size, hidden_size)
        self.w_f = Parameter(torch.tensor(np.random.randn(feature_size + hidden_size, hidden_size)).type(torch.FloatTensor).cuda())
        self.w_i = Parameter(torch.tensor(np.random.randn(feature_size + hidden_size, hidden_size)).type(torch.FloatTensor).cuda())
        self.w_c1 = Parameter(torch.tensor(np.random.randn(feature_size + hidden_size, hidden_size)).type(torch.FloatTensor).cuda())
        self.w_o = Parameter(torch.tensor(np.random.randn(feature_size + hidden_size, hidden_size)).type(torch.FloatTensor).cuda())
        self.state = None
        #self.dropout = TorchNN.Dropout(p=0.7)
        self.full_connection = TorchNN.Linear(hidden_size, num_class).cuda()
        self.sigmod = TorchNN.Sigmoid().cuda()
        self.tanh = TorchNN.Tanh().cuda()
        if bias:
            self.bias_size = (hidden_size)

    def getGradW_f(self):
        logging.basicConfig(level=logging.INFO,
                            filename='grad.log',
                            filemode='w',
                            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        logging.info(self.w_f.grad)
        return self.w_f.grad

    def getGradFull(self):
        for index, param in self.full_connection.named_parameters():
            print(index,' ',param.grad.size())

    def forward(self, data):
        batch_size = data.shape[0]
        sequence_length = data.shape[1]
        if self.state is None:
            h0 = torch.tensor(np.random.randn(self.hidden_size)).type(torch.FloatTensor).cuda()
            c0 = torch.tensor(np.random.randn(self.hidden_size)).type(torch.FloatTensor).cuda()
        else:
            h0, c0 = self.state
        h0 = h0.expand(batch_size, self.hidden_size)
        c0 = c0.expand(batch_size, self.hidden_size)
        for i in range(sequence_length):
            input = data[:,i,:].view(batch_size, -1)
            #input = self.dropout(input)
            #print('input:', input.shape)
            #print('cat:',torch.cat([h0, input], 1))
            f  = self.sigmod(torch.matmul(torch.cat([h0, input], 1).unsqueeze(1), self.w_f))
            i  = self.sigmod(torch.matmul(torch.cat([h0, input], 1).unsqueeze(1), self.w_i))
            c1 = self.tanh(torch.matmul(torch.cat([h0, input], 1).unsqueeze(1), self.w_c1))
            c  = f.squeeze(1) * c0 + i.squeeze(1) * c1.squeeze(1)
            o  = self.sigmod(torch.matmul(torch.cat([h0, input], 1).unsqueeze(1), self.w_o))
            h  = o.squeeze(1) * self.tanh(c)
            h_state = torch.zeros(self.hidden_size).type(torch.FloatTensor).cuda()
            c_state = torch.zeros(self.hidden_size).type(torch.FloatTensor).cuda()
            for h_i in h:
                h_state = h_state + h_i
            for c_i in c:
                c_state = c_state + c_i
            h_state = h_state/batch_size
            c_state = c_state/batch_size
            self.state = (h_state, c_state)
            #self.state = (h[0], c[0])
        result = torch.relu(self.full_connection(h))#.squeeze(1)
        return TorchNNFun.log_softmax(result, dim=1)

class FC(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FC, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_dim, n_hidden_1), torch.nn.BatchNorm1d(n_hidden_1), torch.nn.ReLU(True))
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(n_hidden_1, n_hidden_2), torch.nn.BatchNorm1d(n_hidden_2), torch.nn.ReLU(True))
        self.layer3 = torch.nn.Sequential(torch.nn.Linear(n_hidden_2, out_dim))
        pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class TTRNN(torch.nn.Module):
    def __init__(self, in_modes, out_modes, ranks, nlayers, dropout, rnn_type):
        super(TTRNN, self).__init__()
        in_sizes = int(np.prod(in_modes))
        out_sizes = int(np.prod(out_modes))
        self.prenet = torch.nn.Linear(300, in_sizes)
        self.nlayers = nlayers
        self.dropout = dropout
        self.rnn = torch.nn.ModuleList()
        for ii in range(nlayers):
            if rnn_type == 'ttlstm':
                self.rnn.append(StatefulTTLSTMCell(in_modes if ii == 0 else out_modes, out_modes, ranks))
            elif rnn_type == 'ttgru':
                self.rnn.append(StatefulTTGRUCell(in_modes if ii == 0 else out_modes, out_modes, ranks))
            else:
                raise ValueError()
        self.postnet = torch.nn.Linear(out_sizes, 4).cuda()

    def reset(self):
        for rnn in self.rnn:
            rnn.reset()

    def forward(self, x):
        # x = [batch, max_seq_len, 88] #
        batch, max_seq_len, _ = x.shape
        res = F.leaky_relu(self.prenet(x.view(-1, 300).type(torch.FloatTensor)).cuda().view(batch, max_seq_len, -1), 0.1).cuda()
        list_res = []
        for ii in range(max_seq_len):  # seq_len #
            hidden = res[:, ii].contiguous()
            for jj in range(len(self.rnn)):
                hidden = self.rnn[jj](hidden)
                if isinstance(hidden, (list, tuple)):
                    hidden = hidden[0]
                if self.dropout > 0:
                    hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            list_res.append(hidden)
        res = torch.stack(list_res, dim=1)
        res = self.postnet(res.view(batch * max_seq_len, -1)).view(batch, max_seq_len, -1)  # use last h_t #
        # res = F.sigmoid(res)
        return res
