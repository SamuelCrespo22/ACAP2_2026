import torch
from torch import flatten
from torch import nn

class Generator(nn.Module):
    def __init__(self, inputDim=100, outputDim=512, outputChannels=3):
        super(Generator, self).__init__()

        self.ct1 = nn.ConvTranspose2d(inputDim, 256, 4, 1, 0, bias=False)
        self.relu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(256)

        self.ct2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.relu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(128)

        self.ct3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.relu3 = nn.ReLU()
        self.batchNorm3 = nn.BatchNorm2d(64)

        self.ct4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.relu4 = nn.ReLU()
        self.batchNorm4 = nn.BatchNorm2d(32)

        self.ct5 = nn.ConvTranspose2d(32, outputChannels, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu1(self.batchNorm1(self.ct1(x)))
        x = self.relu2(self.batchNorm2(self.ct2(x)))
        x = self.relu3(self.batchNorm3(self.ct3(x)))
        x = self.relu4(self.batchNorm4(self.ct4(x)))
        output = self.tanh(self.ct5(x))
        return output

class Critic(nn.Module):
    # It is called Critic and does not have a Sigmoid.
    def __init__(self, depth=3, alpha=0.2):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(depth, 32, 4, 2, 1)
        self.leakyRelu1 = nn.LeakyReLU(alpha, inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.leakyRelu2 = nn.LeakyReLU(alpha, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.leakyRelu3 = nn.LeakyReLU(alpha, inplace=True)

        self.fc1 = nn.Linear(8192, 512)
        self.leakyRelu4 = nn.LeakyReLU(alpha, inplace=True)

        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.leakyRelu1(self.conv1(x))
        x = self.leakyRelu2(self.conv2(x))
        x = self.leakyRelu3(self.conv3(x))
        x = flatten(x, 1)
        x = self.leakyRelu4(self.fc1(x))
        output = self.fc2(x)
        return output

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
        