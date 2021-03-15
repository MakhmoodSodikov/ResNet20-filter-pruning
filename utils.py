import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.nn import init as init
from config import CLASSES
from dataloader.cifar10_loader import val_loader
from pruning.filter_pruning import filter_pruning_20
from IPython.display import clear_output


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_inference(model):
    val_load = val_loader()
    data_iter = iter(val_load)
    images, labels = data_iter.next()
    out = model(images.cuda())
    sample_num = random.randint(0, images.shape[0])
    imshow(images[sample_num])

    print('True label: {}'.format(CLASSES[labels[sample_num]]))
    print('Predicted label: {}'.format(CLASSES[torch.argmax(out[sample_num])]))
    print('Accuracy on batch: {}'.format(float(accuracy(out, labels.cuda())[0].cpu().numpy())))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for all of the specified values of k
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def weights_init(m):
    # Custom weight init (kaiming norm)
    # https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
    # '''Is effective to use with deep nets (>30)'''

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def plot_acc_from_k(model_path, init_k=1):
    acc_hist = []
    for k in range(init_k, 32):
        clear_output(True)
        model = filter_pruning_20(model_path, k)
        val_load = val_loader()
        data_iter = iter(val_load)
        images, labels = data_iter.next()
        out = model(images.cuda())
        acc = float(accuracy(out, labels.cuda())[0].cpu().numpy())
        plt.plot(acc_hist)
        plt.set_xlabel('Num clusters')
        plt.set_title('Accuracy')
        plt.show()
        plt.pause(1)
