from __future__ import print_function

import argparse
import os
from tqdm import tqdm

import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

import torch.backends.cudnn as cudnn
cudnn.benchmark = True 

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import datasets
import net


def train(model, train_loader, optimizer, num_devices):
    model.train()
    model_losses = [0]*(num_devices + 1)
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        predictions = model(data)
        total_loss = 0
        for i, prediction in enumerate(predictions):
            loss = F.cross_entropy(prediction, target)
            model_losses[i] += loss.sum()*len(target)
            total_loss += loss
        total_loss.backward()
        optimizer.step()

    N = len(train_loader.dataset)
    loss_str = ', '.join(['dev-{}: {:.4f}'.format(i, loss.data[0] / N)
                        for i, loss in enumerate(model_losses[:-1])])
    print('Train Loss:: {}, cloud-{:.4f}'.format(loss_str, model_losses[-1].data[0] / N))

    return model_losses
 

def test(model, test_loader, num_devices):
    model.eval()
    model_losses = [0]*(num_devices + 1)
    num_correct = [0]*(num_devices + 1)
    for data, target in tqdm(test_loader, leave=False):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        predictions = model(data)
        for i, prediction in enumerate(predictions):
            loss = F.cross_entropy(prediction, target, size_average=False).data[0]
            pred = prediction.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
            num_correct[i] += correct
            model_losses[i] += loss

    N = len(test_loader.dataset)
    loss_str = ', '.join(['dev-{}: {:.4f}'.format(i, loss / N)
                        for i, loss in enumerate(model_losses[:-1])])
    acc_str = ', '.join(['dev-{}: {:.4f}%'.format(i, 100. * (correct / N))
                        for i, correct in enumerate(num_correct[:-1])])
    print('Test  Loss:: {}, cloud-{:.4f}'.format(loss_str, model_losses[-1] / N))
    print('Test  Acc.:: {}, cloud-{:.4f}'.format(acc_str, 100. * (num_correct[-1] / N)))

    return model_losses, num_correct


def train_model(model, model_path, train_loader, test_loader, lr, epochs, num_devices):
    ps = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.SGD(ps, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(1, epochs):
        print('[Epoch {}]'.format(epoch))
        train(model, train_loader, optimizer, num_devices)
        test(model, test_loader, num_devices)
        torch.save(model, model_path)
        scheduler.step()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Example')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output', default='models/model.pth',
                        help='output directory')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    train_dataset, train_loader, test_dataset, test_loader = data
    x, _ = train_loader.__iter__().next()
    num_devices = x.shape[1]
    in_channels = x.shape[2]
    model = net.DDNN(in_channels, 10, num_devices)
    train_model(model, args.output, train_loader, test_loader, args.lr, args.epochs, num_devices)