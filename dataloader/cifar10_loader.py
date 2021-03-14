import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config import *


def train_loader():
    return torch.utils.data.DataLoader(
                        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, 4),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                        ]), download=True),
                        batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=LOADER_NUM_WORKERS, pin_memory=True)


def val_loader():
    return torch.utils.data.DataLoader(
                        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                        ])),
                        batch_size=128, shuffle=False,
                        num_workers=LOADER_NUM_WORKERS, pin_memory=True)
