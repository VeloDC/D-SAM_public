import torch
import torch.nn as nn
from torch.nn import ModuleList as ML
from torchvision.models import resnet

__all__ = ['DSAM_resnet18', 'deepall_resnet18']


def DSAM_resnet18(num_classes=1000, pretrained=True, num_domains=3, batch_size=32):
    if pretrained:
        print('initializing from pretrained')
    pretrained = resnet.resnet18(pretrained=pretrained)
    model = DSAM_ResNet18(pretrained, num_classes=num_classes, num_domains=num_domains, batch_size=batch_size)
    return model


def deepall_resnet18(num_classes=1000, pretrained=True, num_domains=3, batch_size=32):
    pretrained = resnet.resnet18(pretrained=pretrained)
    model = DeepAll_ResNet18(pretrained, num_classes=num_classes)
    return model


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CAL(nn.Module):
    def __init__(self, low_in_channels, same_in_channels, out_channels, requires_downsample=False):
        super(CAL, self).__init__()

        self.requires_downsample = requires_downsample
        self.aggregation_layer = nn.Sequential(nn.Dropout2d(),
                                               nn.Conv2d(low_in_channels + same_in_channels, out_channels, kernel_size=1),
                                               nn.ReLU(inplace=True))
        
        if requires_downsample:
            self.downsampling_layer = nn.Sequential(nn.Conv2d(low_in_channels, low_in_channels, kernel_size=1, stride=2),
                                                    nn.ReLU(inplace=True))

    def forward(self, x):
        low, same = x
        if self.requires_downsample:
            low = self.downsampling_layer(low)
        cat = torch.cat((low, same), 1)
        aggregation = self.aggregation_layer(cat)
        return aggregation


class Convolutional_Aggregation_Module(nn.Module):
    def __init__(self, num_classes):
        super(Convolutional_Aggregation_Module, self).__init__()
        n = 2
        self.a_1 = CAL(64, 64, 64 * n)
        
        self.a_2 = CAL(64 * n, 128, 128 * n, True)
        self.a_3 = CAL(128*n, 128, 128*n)

        self.a_4 = CAL(128*n, 256, 256*n, True)
        self.a_5 = CAL(256*n, 256, 256*n)

        self.a_6 = CAL(256*n, 512, 512*n, True)
        self.a_7 = CAL(512*n, 512, 512*n)

        self.flatten = nn.Sequential(nn.AvgPool2d(7, stride=1),
                                     Flatten())
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(512*n, num_classes))

    def forward(self, x):
        l1_0, l1_1, l2_0, l2_1, l3_0, l3_1, l4_0, l4_1 = x

        x = self.a_1((l1_0, l1_1))
        x = self.a_2((x, l2_0))
        x = self.a_3((x, l2_1))
        x = self.a_4((x, l3_0))
        x = self.a_5((x, l3_1))
        x = self.a_6((x, l4_0))
        x = self.a_7((x, l4_1))

        out = self.flatten(x)
        out = self.classifier(out)

        return out


class DSAM_ResNet18(nn.Module):
    def __init__(self, net, num_classes=1000, num_domains=3, batch_size=32, aggr_fn=Convolutional_Aggregation_Module):
        super(DSAM_ResNet18, self).__init__()

        self.num_domains = num_domains
        self.bs = batch_size

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1_0 = net.layer1[0]
        self.layer1_1 = net.layer1[1]

        self.layer2_0 = net.layer2[0]
        self.layer2_1 = net.layer2[1]

        self.layer3_0 = net.layer3[0]
        self.layer3_1 = net.layer3[1]

        self.layer4_0 = net.layer4[0]
        self.layer4_1 = net.layer4[1]

        self.aggregation_modules = ML([aggr_fn(num_classes) for _ in range(num_domains)])

    def main_pass(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1_0 = self.layer1_0(x)
        l1_1 = self.layer1_1(l1_0)

        l2_0 = self.layer2_0(l1_1)
        l2_1 = self.layer2_1(l2_0)

        l3_0 = self.layer3_0(l2_1)
        l3_1 = self.layer3_1(l3_0)

        l4_0 = self.layer4_0(l3_1)
        l4_1 = self.layer4_1(l4_0)

        return l1_0,l1_1,l2_0,l2_1,l3_0,l3_1,l4_0,l4_1


    def forward(self, x):        
        if len(x) == 2:
            x, i = x
            main_outs = self.main_pass(x)

            output = self.aggregation_modules[i](main_outs)
            return output

        else:
            main_outs = self.main_pass(x)
            outs = [self.aggregation_modules[i]((main_outs[0][(i*self.bs):((i+1)*self.bs)],
                                                 main_outs[1][(i*self.bs):((i+1)*self.bs)],
                                                 main_outs[2][(i*self.bs):((i+1)*self.bs)],
                                                 main_outs[3][(i*self.bs):((i+1)*self.bs)],
                                                 main_outs[4][(i*self.bs):((i+1)*self.bs)],
                                                 main_outs[5][(i*self.bs):((i+1)*self.bs)],
                                                 main_outs[6][(i*self.bs):((i+1)*self.bs)],
                                                 main_outs[7][(i*self.bs):((i+1)*self.bs)])) for i in range(self.num_domains)]

            return outs


class DeepAll_ResNet18(nn.Module):
    def __init__(self, net, num_classes=1000):
        super(DeepAll_ResNet18, self).__init__()

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.avgpool = net.avgpool
        self.fc = nn.Linear(512, num_classes)

    def main_pass(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x):
        if len(x) == 2:
            x, _ = x

            out = self.main_pass(x)
            return out

        else:
            out = self.main_pass(x)

            return [out]
