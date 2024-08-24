import torch
from torch import nn


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, image_size, conv_dim=64, channels=3, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        layers = []

        layers.append(nn.Linear(image_size + c_dim, 128))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(128, 128))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(128, image_size))

        layers.append(nn.ReLU())

        # layers.append(nn.Tanh())  # changed from tanh
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # c = torch.argmax(c, dim=-1).unsqueeze(1)
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, channels=3, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []

        layers.append(nn.Linear(image_size, 256))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(256, 256))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

        self.conv1 = nn.Linear(256, 1)
        self.conv2 = nn.Linear(256, c_dim)

    def forward(self, x):
        h = self.main(x)
        out_src = torch.sigmoid(self.conv1(h))
        out_cls = self.conv2(h)
        return out_src, out_cls