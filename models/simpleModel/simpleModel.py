import torch 
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(in_size, 512)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(1024, 2048)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(2048, 1024)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Linear(1024, 512)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Linear(512, out_size)
        self.layer12 = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        return x
    
