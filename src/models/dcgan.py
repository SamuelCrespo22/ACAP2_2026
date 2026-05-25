import torch
from torch import flatten
from torch import nn

class Generator(nn.Module):
    def __init__(self, inputDim=100, outputDim=512, outputChannels=3):
        super(Generator, self).__init__()

        # (100, 1, 1) -> (256, 4, 4).
        self.ct1 = nn.ConvTranspose2d(inputDim, 256, 4, 1, 0, bias=False)
        self.relu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(256)

        # (256, 4, 4) -> (128, 8, 8).
        self.ct2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.relu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(128)

        # (128, 8, 8) -> (64, 16, 16).
        self.ct3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.relu3 = nn.ReLU()
        self.batchNorm3 = nn.BatchNorm2d(64)

        # (64, 16, 16) -> (32, 32, 32).
        self.ct4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.relu4 = nn.ReLU()
        self.batchNorm4 = nn.BatchNorm2d(32)

        # (32, 32, 32) -> (3, 64, 64).
        self.ct5 = nn.ConvTranspose2d(32, outputChannels, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.ct1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.ct2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.ct3(x)
        x = self.batchNorm3(x)
        x = self.relu3(x)
        x = self.ct4(x)
        x = self.batchNorm4(x)
        x = self.relu4(x)
        output = self.tanh(self.ct5(x))
        return output

class Discriminator(nn.Module):
    def __init__(self, depth=3, alpha=0.2):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(depth, 32, 4, 2, 1)
        self.leakyRelu1 = nn.LeakyReLU(alpha, inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.leakyRelu2 = nn.LeakyReLU(alpha, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.leakyRelu3 = nn.LeakyReLU(alpha, inplace=True)

        # For 64x64 images, after three convs of stride 2, we get 8x8.
        # 128 channels * 8 * 8 = 8192.
        self.fc1 = nn.Linear(8192, 512)
        self.leakyRelu4 = nn.LeakyReLU(alpha, inplace=True)

        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.leakyRelu1(self.conv1(x))
        x = self.leakyRelu2(self.batchNorm2(self.conv2(x)))
        x = self.leakyRelu3(self.batchNorm3(self.conv3(x)))
        x = flatten(x, 1)
        x = self.leakyRelu4(self.fc1(x))
        output = self.sigmoid(self.fc2(x))
        return output

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)