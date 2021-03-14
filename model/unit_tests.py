import numpy as np
from model.resnet20 import ResNet20


def smoke_test(net):
    print('---Smoke test')
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))
    print('---')


def layers_check(net):
    print('---Layers check')

    print(net.conv1.weight.size())
    for l in net.layer1:
        print(l.conv1.weight.size())
    for l in net.layer2:
        print(l.conv1.weight.size())
    for l in net.layer3:
        print(l.conv1.weight.size())
    print('---')


if __name__ == '__main__':
    net = ResNet20()
    smoke_test(net)
    layers_check(net)
