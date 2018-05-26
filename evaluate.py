import argparse
import torch
from tqdm import tqdm

import datasets
from torch.autograd import Variable
import torch.nn.functional as F

def test_outage(model, test_loader, num_devices, outages):
    model.eval()
    num_correct = 0
    for data, target in tqdm(test_loader, leave=False):
        for outage in outages:
            data[:, outage] = 0
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        predictions = model(data)
        cloud_pred = predictions[-1]
        loss = F.cross_entropy(cloud_pred, target, size_average=False).item()
        pred = cloud_pred.data.max(1, keepdim=True)[1]
        correct = (pred.view(-1) == target.view(-1)).long().sum().item()
        num_correct += correct

    N = len(test_loader.dataset)

    return 100. * (num_correct / N)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='models/model.pth',
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
    model = torch.load(args.model_path)
    for i in range(num_devices):
        outages = [i]
        acc = test_outage(model, test_loader, num_devices, outages)
        print('Missing Device(s) {}: {:.4f}'.format(outages, acc))

    for i in range(1, num_devices + 1):
        outages = list(range(i, num_devices))
        acc = test_outage(model, test_loader, num_devices, outages)
        print('Missing Device(s) {}: {:.4f}'.format(outages, acc))