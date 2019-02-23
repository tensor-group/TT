import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import modules.ttrnn as ttrnn
class RNN(nn.Module):

    def __init__(self,input_size,hidden_size,ranks,dropout,num_classes):
        super(RNN,self).__init__()
        in_sizes = int(np.prod(input_size))
        out_sizes = int(np.prod(hidden_size))
        #self.input_size = input_size
        #self.hidden_size = hidden_size
        #self.num_layers = num_layers
        #self.rnn = nn.GRU(input_size,hidden_size,num_layers,batch_first=True,dropout=0.7)#,nonlinearity='relu')
        #self.cell = nn.GRUCell(input_size,hidden_size).cuda()
        #self.cpcell = cprnn.StatefulCPGRUCell([1,1,1],[4,4,8],3).cuda()
        self.ttcell = ttrnn.StatefulTTGRUCell(input_size,hidden_size,ranks).cuda()
        self.dropout = nn.Dropout(p=dropout).cuda()
        #self.batchnorm = nn.BatchNorm1d(hidden_size).cuda()
        self.fc = nn.Linear(out_sizes,num_classes).cuda()
        self.activ = nn.ReLU().cuda()

    def forward(self, input, length=None):
        # if length is None:
        #     length = input.size(1)
        # h_t = Variable(torch.zeros(input.size(0), self.hidden_size).cuda())
        # outputs = []
        # for i in range(length):
        #     #replace the following with multi-layer cell
        #     x = input[:,i,:].view(input.size(0), -1)
        #     x = self.dropout(x)
        #     #h_t = self.batchnorm(h_t)
        #     cellout = self.cell(x,h_t)
        #     h_t = cellout
        #     outputs.append(cellout)
        # o = self.fc(outputs[-1])
        # ot = self.activ(o)
        #if length is None:
        #    length = input.size(1)
        outputs = []
        for i in length: #length.shape=(batch_size, sequence_length)
            for j in range(i.item()):
                x = input[:, j, :].view(input.size(0), -1)
                x = self.dropout(x)
                out = self.ttcell(x)
                outputs.append(out) 
        ot = self.fc(outputs[-1])
        ot = self.activ(ot)
        return ot
