import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

__all__ = [
    "ResNet",
    "resnet18"
]


class ShortcutA(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def __call__(self, x):
        return mx.pad(
            x[:, ::2, ::2, :],
            pad_width=[(0, 0), (0, 0), (0, 0), (self.dims // 4, self.dims // 4)],
        )


class Block(nn.Module):
    """
    Implements a ResNet block with two convolutional layers and a skip connection.
    As per the paper, CIFAR-10 uses Shortcut type-A skip connections. (See paper for details)
    """

    def __init__(self, in_dims, dims, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_dims, 
            dims, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm(dims)

        self.conv2 = nn.Conv2d(
            dims, 
            dims, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm(dims)

        if stride != 1:
            self.shortcut = ShortcutA(dims)
        else:
            self.shortcut = None

    def __call__(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut is None:
            out += x
        else:
            out += self.shortcut(x)
        
        out = nn.relu(out)
        return out


class ResNet(nn.Module):
    """
    Creates a ResNet model for CIFAR-10, as specified in the original paper.
    """

    def __init__(self, 
                 img_channels,
                 num_layers,
                 block,
                 num_classes=1000):
        super().__init__()
        
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels = img_channels, 
                               out_channels=self.in_channels, 
                               kernel_size=7,
                               stride=2, 
                               padding=3, 
                               bias=False)
        
        self.bn1 = nn.BatchNorm(self.in_channels)
        
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, 
            stride=2, 
            padding=1)

        self.layer1 = self._make_layer(block, 
                                       64, 
                                       layers[0])
        
        self.layer2 = self._make_layer(block, 
                                       128, 
                                       layers[1], 
                                       stride=2)
        
        self.layer3 = self._make_layer(block, 
                                       256, 
                                       layers[2], 
                                       stride=2)
        
        self.layer4 = self._make_layer(block, 
                                       512, 
                                       layers[3], 
                                       stride=2)
        
        self.avgpool = nn.AvgPool2d((1, 1))

        self.linear = nn.Linear(512, 
                                num_classes)

    def _make_layer(self, 
                    block, 
                    dims, 
                    layers, 
                    stride=1):
        layers = []
        
        layers.append(
            block(self.in_channels, dims, stride)
        )
        
        self.in_channels = dims
        
        for _ in range(1):
            layers.append(
                block(self.in_channels, dims)
            )
        
        return nn.Sequential(*layers)

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.parameters()))
        return nparams

    def __call__(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = mx.flatten(x, 1)
        x = self.linear(x)
        
        return x
        
def resnet18(**kwargs):
    return ResNet(img_channels=3, num_layers=18, block=Block, num_classes=1000, **kwargs)