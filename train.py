from model.resnet20 import ResNet20
import torch
import torch.nn as nn
from config import *
from dataloader.cifar10_loader import train_loader, val_loader
import time
from utils import accuracy, AverageMeter, save_model, show_plots

__all__ = ['train', 'validate']


def train(model, epochs=EPOCHS, lr=INIT_LR, required_precision=None):

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.99, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES)
    criterion = nn.CrossEntropyLoss().cuda()

    prec_hist = {'train': [],
                 'val': []}

    loss_hist = {'train': [],
                 'val': []}

    _train_loader = train_loader()
    _val_loader = val_loader()

    for epoch in range(0, epochs):

        # train for one epoch
        train_epoch(_train_loader, model, criterion, optimizer, epoch, prec_hist, loss_hist)
        lr_scheduler.step()

        # evaluate on validation set
        validate(_val_loader, model, criterion, prec_hist, loss_hist)

        # save checkpoint and show plots
        save_model(required_precision, epoch, model, prec_hist['val'][-1])
        show_plots(optimizer.param_groups[0]['lr'], prec_hist, loss_hist)


def validate(_val_loader, model, criterion, prec_hist, loss_hist):
    """
    Run evaluation
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (inp, target) in enumerate(_val_loader):
            target = target.cuda()
            input_var = inp.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)[0]
            losses.update(loss.item(), inp.size(0))
            top1.update(acc.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    loss_hist['val'].append(losses.avg)
    prec_hist['val'].append(top1.avg)


def train_epoch(_train_loader, model, criterion, optimizer, epoch, prec_hist, loss_hist):
    """
    Run training on one epoch
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inp, target) in enumerate(_train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = inp.cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var).cuda()
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        acc = accuracy(output.data, target)[0]
        losses.update(loss.item(), inp.size(0))
        top1.update(acc.item(), inp.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    loss_hist['train'].append(losses.avg)
    prec_hist['train'].append(top1.avg)


def __main__():
    net = ResNet20()
    epochs = EPOCHS
    train(net, epochs)


if __name__ == '__main__':
    __main__()
