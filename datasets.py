import os

from torchvision import datasets, transforms
import torch
from util import Partition 


class FashionMNIST(datasets.MNIST):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

def get_dataset(dataset_root, dataset, batch_size, is_cuda=True):
    if dataset == 'mnist':
        train, train_loader, test, test_loader = get_mnist(dataset_root, batch_size, is_cuda)
    elif dataset == 'fashion-mnist':
        train, train_loader, test, test_loader = get_fashion_mnist(dataset_root, batch_size, is_cuda)
    elif dataset == 'cifar10':
        train, train_loader, test, test_loader = get_cifar10(dataset_root, batch_size, is_cuda)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))

    return train, train_loader, test, test_loader

def get_mnist(dataset_root, batch_size, is_cuda=True):
    kwargs = {'num_workers': 12, 'pin_memory': True} if is_cuda else {}
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True, download=True, 
                        transform=transforms.Compose([
                            transforms.Pad(2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            Partition(3, 2),
                        ]))
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            Partition(3, 2),
                        ]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=True, **kwargs)
    
    return train, train_loader, test, test_loader

def get_fashion_mnist(dataset_root, batch_size, is_cuda=True):
    kwargs = {'num_workers': 12, 'pin_memory': True} if is_cuda else {}
    train = FashionMNIST(os.path.join(dataset_root, 'fashion_mnist'), train=True, download=True, 
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            Partition(3, 2),
                        ]))
    test = FashionMNIST(os.path.join(dataset_root, 'fashion_mnist'), train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.Pad(2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            Partition(3, 2),
                        ]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=False, **kwargs)
    
    return train, train_loader, test, test_loader


def get_cifar10(dataset_root, batch_size, is_cuda=True):
    kwargs = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                            Partition(3, 2),
                        ]))
    test = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                            Partition(3, 2),
                        ]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=False, **kwargs)
 
    return train, train_loader, test, test_loader
