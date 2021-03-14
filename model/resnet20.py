from model.src import ResNet, BasicBlock


__all__ = ['ResNet20']


def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])


