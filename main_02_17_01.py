import torch
import torchviz
import argparse
import torchvision
import numpy as np
from Net import LSTM, FC, Net
import torch.nn as TorchNN
import torch.optim as TorchOptim
import torch.utils.data as TorchData
import torch.autograd as TorchAutograd
import torch.nn.functional as TorchNNFun

parser = argparse.ArgumentParser(description='Python Mnist Example')
parser.add_argument('-train_batch_size', type=int, default=64, metavar='--train-batch',
                    help='input batch size for training(default:64)')
parser.add_argument('-test_batch_size', type=int, default=1000, metavar='--test-batch',
                    help='input batch size for testing(default:1000)')
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
    dataset = torchvision.datasets.MNIST('./data',
                                         train=True,
                                         download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0,), (1,))
                                         ])),
    batch_size=args.train_batch_size,
    drop_last=True,
    shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    dataset = torchvision.datasets.MNIST('./data',
                                         train=False,
                                         download=True,
                                         transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0,), (1,))
                                         ])),
    batch_size=args.test_batch_size,
    shuffle=True,
    drop_last=True,
    **kwargs
)
model = LSTM(28*28, 64, 10)
#model = FC(28 * 28, 300, 100, 10)
model = Net(28*28, 64, )
if args.cuda:
    model.cuda()
optimizer = TorchOptim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    #model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = TorchAutograd.Variable(data), TorchAutograd.Variable(target)
        optimizer.zero_grad()
        output = model(data.view(args.train_batch_size, -1, 28*28))
        #output = model(data.view(args.train_batch_size, -1))
        #pred = output.data.max(1, keepdim=True)[1]
        #print(pred)
        loss = TorchNNFun.cross_entropy(output, target)
        loss.backward(retain_graph=True)
        model.getGradW_f()
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
        output = model(data.view(args.test_batch_size, -1, 28*28))
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