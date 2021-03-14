from model.resnet20 import ResNet20
import torch
import torch.nn as nn
from config import *
from dataloader.cifar10_loader import train_loader, val_loader
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils import accuracy, AverageMeter


__all__ = ['train', 'validate']


def train(model, epochs=EPOCHS, lr=INIT_LR, required_precision=None):

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.99, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES)
    criterion = nn.CrossEntropyLoss().cuda()

    prec_hist = {'train': [], 'val': []}
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
        # prec_hist.append(prec)

        # save checkpoint and show plots
        save_model(required_precision, epoch, model, prec_hist['val'][-1])
        show_plots(optimizer.param_groups[0]['lr'], prec_hist, loss_hist)


def save_model(required_precision, epoch, model, precision):

    if epoch > 0 and epoch % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, 'checkpoints/checkpoint_at_{}_epoch.th'.format(epoch))

    if precision >= required_precision and required_precision:
        torch.save({
            'state_dict': model.state_dict(),
            'precision': precision,
        }, 'models/best_model_{}.th'.format(precision))


def show_plots(lr, prec_hist, loss_hist):

    clear_output(True)
    print('current lr {:.5e}'.format(lr))
    print('Precision @ k {:.3f}'.format(prec_hist[-1]))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    ax[0].plot(loss_hist['train'], label='train loss')
    ax[0].plot(loss_hist['val'], label='validation loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_title('Train loss')

    ax[1].plot(prec_hist['train'], label='train accuracy')
    ax[1].plot(prec_hist['val'], label='validation accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_title('Train accuracy')

    plt.figure(figsize=(10, 12))
    plt.pause(1)
    # plt.subplot(211)
    # plt.gca().set_title('Loss on epoch')
    # plt.plot(loss_hist['train'], label='Train')
    # plt.plot(loss_hist['val'], label='Val')
    # plt.subplot(212)
    # plt.gca().set_title('Accuracy at top@1 on epoch')
    # plt.plot(prec_hist)


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

            # if i % 5 == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #               i, len(_val_loader), batch_time=batch_time, loss=losses,
            #               top1=top1))

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

    # print('Epoch: [{0}][{1}/{2}]\t'
    #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
    #           epoch, i, len(_train_loader), batch_time=batch_time,
    #           data_time=data_time, loss=losses, top1=top1))


def __main__():
    net = ResNet20()
    epochs = EPOCHS
    train(net, epochs)


if __name__ == '__main__':
    __main__()
