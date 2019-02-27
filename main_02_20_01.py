import torch
import torchviz
import argparse
import DataUtils
import torchvision
import numpy as np
import torch.nn as TorchNN
from Net import LSTM, FC, TTRNN
from RNN import RNN
import torch.optim as TorchOptim
import torch.utils.data as TorchData
import torch.autograd as TorchAutograd
import torch.nn.functional as TorchNNFun

train_path = '/home/hkw/training2017'
test_path = '/home/hkw/validation'

parser = argparse.ArgumentParser(description='ECG TTRNN')
parser.add_argument('-train_batch_size', type=int, default=64, metavar='--train-batch',
                    help='input batch size for training(default:64)')
parser.add_argument('-test_batch_size', type=int, default=10, metavar='--test-batch',
                    help='input batch size for testing(default:1000)')
parser.add_argument('-feature_size', type=int, default=300, metavar='--test-batch',
                    help='feature_size')
parser.add_argument('-epochs', type=int, default=10, metavar='--epochs',
                    help='number of epoch to train(default:10)')
parser.add_argument('-lr', type=float, default=0.01, metavar='--learning-rate',
                    help='learning rate(default:0.01)')
parser.add_argument('-m', type=float, default=0.5, metavar='--momentum',
                    help='SGD momentum(default:0.5)')
parser.add_argument('-no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('-seed', type=int, default=1, metavar='S',
                    help='random seed(default:1)')
parser.add_argument('-log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 3, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    DataUtils.ECGDataset(train_path,test_path,train=True),
    batch_size=args.train_batch_size,
    drop_last=True,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    DataUtils.ECGDataset(train_path,test_path,test=True),
    batch_size=args.test_batch_size,
    shuffle=True,
    drop_last=True
)
#model = LSTM(28*28, 64, 10)
#model = FC(28 * 28, 300, 100, 10)
#model = TTRNN([4,7,4,7], [4,2,4,4], [1,3,4,2,1], 1, 0.8, 'ttgru')
model = RNN([2,5,2,7], [4,2,4,4], [1,3,4,2,1],0.8,4)
if args.cuda:
    model.cuda()
optimizer = TorchOptim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    #model.train()
    for batch_idx, data in enumerate(train_loader):
        train = data[0]
        target = data[1]
        sequence_length = data[2]
        if args.cuda:
            data, target = train.cuda(), target.cuda()
        #data, target = TorchAutograd.Variable(data), TorchAutograd.Variable(target)
        output = model(data.view(args.train_batch_size, -1, args.feature_size).float(), length=sequence_length)
        #output = model(data.view(args.train_batch_size, -1))
        #pred = output.data.max(1, keepdim=True)[1]
        #print(pred)
        loss = TorchNNFun.cross_entropy(output, target)
        loss.backward(retain_graph=True)
        #model.getGradW_f()
        optimizer.zero_grad()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = TorchAutograd.Variable(data), TorchAutograd.Variable(target)
        output = model(data.view(args.test_batch_size, -1, args.feature_size))
        #output = model(data.view(args.test_batch_size, -1))
        test_loss += TorchNNFun.cross_entropy(output,target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, args.test_batch_size,
        100. * correct / args.test_batch_size))

def run():
    torch.multiprocessing.freeze_support()
    torch.cuda.set_device(0)

if __name__ == '__main__':
    run()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()