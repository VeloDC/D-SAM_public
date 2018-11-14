import torch
import torch.nn as nn
from torch.nn import ModuleList as ML
from torchvision.models import alexnet

__all__ = ['deepall_alexnet', 'DSAM_alexnet']

ch = {
    'conv1': 64,
    'conv2': 192,
    'conv3': 384,
    'conv4': 256,
    'conv5': 256,
    'flatten': 256*6*6,
    'fc7': 4096
}


def deepall_alexnet(num_classes=1000, pretrained=True, num_domains=3, batch_size=32):
    pretrained = alexnet(pretrained=pretrained)
    model = DeepAll_AlexNet(pretrained, num_classes)
    return model


def DSAM_alexnet(num_classes=1000, pretrained=True, num_domains=3, batch_size=32):
    pretrained = alexnet(pretrained=pretrained)
    model = DSAM_AlexNet(pretrained, num_classes, num_domains, batch_size=batch_size)
    return model


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class FCAL(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCAL, self).__init__()

        self.aggregation_layer = nn.Sequential(nn.ReLU(),
                                               nn.Linear(in_channels, out_channels),
                                               nn.ReLU(inplace=True))

    def forward(self, x):
        x = torch.cat(x, 1)
        aggregation = self.aggregation_layer(x)
        return aggregation


class CAL(nn.Module):
    def __init__(self, low_in_channels, same_in_channels, out_channels, requires_downsample=False):
        super(CAL, self).__init__()

        self.requires_downsample = requires_downsample
        self.aggregation_layer = nn.Sequential(nn.ReLU(),
                                               nn.Conv2d(low_in_channels + same_in_channels, out_channels, kernel_size=1),
                                               nn.ReLU(inplace=True))
        if requires_downsample:
            self.downsampling_layer = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2))

    def forward(self, x):
        low, same = x
        if self.requires_downsample:
            low = self.downsampling_layer(low)
        cat = torch.cat((low, same), 1)
        aggregation = self.aggregation_layer(cat)
        return aggregation


class Aggregation_Module(nn.Module):
    def __init__(self, num_classes):
        super(Aggregation_Module, self).__init__()

        self.aggregation_1 = CAL(ch['conv1'], ch['conv2'], ch['conv2'], True)
        self.aggregation_2 = CAL(ch['conv2'], ch['conv3'], ch['conv3'], True)
        self.aggregation_3 = CAL(ch['conv3'], ch['conv4'], ch['conv4'])
        self.aggregation_4 = CAL(ch['conv4'], ch['conv5'], ch['conv5'])

        self.flatten = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2),
                                     Flatten())
        self.aggregation_5 = FCAL(ch['flatten'] + ch['fc7'], 512)
        self.aggregation_6 = FCAL(4096 + 512, 512)
        
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(512, num_classes))

    def forward(self, x):
        c1, c2, c3, c4, c5, fc6, fc7 = x

        aggr1 = self.aggregation_1((c1,c2))
        aggr2 = self.aggregation_2((aggr1, c3))
        aggr3 = self.aggregation_3((aggr2, c4))
        aggr4 = self.aggregation_4((aggr3, c5))
        aggr4 = self.flatten(aggr4)
        aggr5 = self.aggregation_5((aggr4, fc6))
        aggr6 = self.aggregation_6((aggr5, fc7))
        out = self.classifier(aggr6)
        return aggr6


class DSAM_AlexNet(nn.Module):
    def __init__(self, net, num_classes=None, num_domains=3, batch_size=32):
        super(DSAM_AlexNet, self).__init__()

        self.num_domains = num_domains
        self.bs = batch_size

        self.conv1 = nn.Sequential(*list(net.features.children())[0:1])
        self.conv2 = nn.Sequential(*list(net.features.children())[1:4])
        self.conv3 = nn.Sequential(*list(net.features.children())[4:7])
        self.conv4 = nn.Sequential(*list(net.features.children())[7:9])
        self.conv5 = nn.Sequential(*list(net.features.children())[9:11])
        self.pool5 = nn.Sequential(*list(net.features.children())[11:13])
        self.flatten = nn.Sequential(Flatten())
        self.fc6 = nn.Sequential(*list(net.classifier.children())[0:2])
        self.fc7 = nn.Sequential(*list(net.classifier.children())[2:5])

        self.aggregation_modules = ML([Aggregation_Module(num_classes) for _ in range(num_domains)])

    def forward(self, x):
        if len(x)==2:
            x, i = x
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            pool5 = self.pool5(conv5)
            flatten = self.flatten(pool5)
            fc6 = self.fc6(flatten)
            fc7 = self.fc7(fc6)
            output = self.aggregation_modules[i]((conv1, conv2, conv3, conv4, conv5, fc6, fc7))
            return output

        else:
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            pool5 = self.pool5(conv5)
            flatten = self.flatten(pool5)
            fc6 = self.fc6(flatten)
            fc7 = self.fc7(fc6)
            
            outs = [self.aggregation_modules[i]((conv1[(i*self.bs):((i+1)*self.bs)],
                                                 conv2[(i*self.bs):((i+1)*self.bs)],
                                                 conv3[(i*self.bs):((i+1)*self.bs)],
                                                 conv4[(i*self.bs):((i+1)*self.bs)],
                                                 conv5[(i*self.bs):((i+1)*self.bs)],
                                                 fc6[(i*self.bs):((i+1)*self.bs)],
                                                 fc7[(i*self.bs):((i+1)*self.bs)]))
                    for i in range(self.num_domains)]
            return outs


class DeepAll_AlexNet(nn.Module):
    def __init__(self, net, num_classes=None):
        super(DeepAll_AlexNet, self).__init__()
        
        self.conv1 = nn.Sequential(*list(net.features.children())[0:1])
        self.conv2 = nn.Sequential(*list(net.features.children())[1:4])
        self.conv3 = nn.Sequential(*list(net.features.children())[4:7])
        self.conv4 = nn.Sequential(*list(net.features.children())[7:9])
        self.conv5 = nn.Sequential(*list(net.features.children())[9:11])
        self.pool5 = nn.Sequential(*list(net.features.children())[11:13])
        self.flatten = nn.Sequential(Flatten())
        self.fc6 = nn.Sequential(*list(net.classifier.children())[0:2])
        self.fc7 = nn.Sequential(*list(net.classifier.children())[2:5])
        self.classifier = nn.Sequential(nn.ReLU(inplace=True),
                                        nn.Linear(4096, num_classes))

    def forward(self, x):
        if len(x)==2:
            x, _ = x
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            pool5 = self.pool5(conv5)
            flatten = self.flatten(pool5)
            fc6 = self.fc6(flatten)
            fc7 = self.fc7(fc6)
            return self.classifier(fc7)

        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.pool5(x)
            x = self.flatten(x)
            x = self.fc6(x)
            x = self.fc7(x)
            return [self.classifier(x)]
