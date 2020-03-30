import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        #ENCODER
        
        self.convE1 = nn.Conv2d(1, 64, kernel_size = (3, 3), stride = 1, padding = 1)#512x512
        self.convE2 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 1, padding = 1)#512x512
        self.convE3 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = 1, padding = 1)#256x256
        self.convE4 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 1, padding = 1)#256x256
        self.convE5 = nn.Conv2d(128, 256, kernel_size = (3, 3), stride = 1, padding = 1)#128x128
        self.convE6 = nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 1, padding = 1)#128x128
        self.convE7 = nn.Conv2d(256, 512, kernel_size = (3, 3), stride = 1, padding = 1)#64x64
        self.convE8 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = 1, padding = 1)#64x64
        self.convE9 = nn.Conv2d(512, 1024, kernel_size = (3, 3), stride = 1, padding = 1)#32x32
        self.convE10 = nn.Conv2d(1024, 1024, kernel_size = (3, 3), stride = 1, padding = 1)#32x32
        
        #DECODER
        
        self.convT1 = nn.ConvTranspose2d(1024, 512, kernel_size = (4, 4), stride = 2, padding = 1)#64x64
        #concat
        self.convD1 = nn.Conv2d(1024, 512, kernel_size = (3, 3), stride = 1, padding = 1)#64x64
        self.convD2 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = 1, padding = 1)#64x64
        
        self.convT2 = nn.ConvTranspose2d(512, 256, kernel_size = (4, 4), stride = 2, padding = 1)#128x128
        #concat
        self.convD3 = nn.Conv2d(512, 256, kernel_size = (3, 3), stride = 1, padding = 1)#128x128
        self.convD4 = nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 1, padding = 1)#128x128
        
        self.convT3 = nn.ConvTranspose2d(256, 128, kernel_size = (4, 4), stride = 2, padding = 1)#256x256
        #concat
        self.convD5 = nn.Conv2d(256, 128, kernel_size = (3, 3), stride = 1, padding = 1)#256x256
        self.convD6 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 1, padding = 1)#256x256
        
        self.convT4 = nn.ConvTranspose2d(128, 64, kernel_size = (4, 4), stride = 2, padding = 1)
        #concat
        self.convD7 = nn.Conv2d(128, 64, kernel_size = (3, 3), stride = 1, padding = 1)#512x512
        self.convD8 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 1, padding = 1)#512x512
        self.convD9 = nn.Conv2d(64, 1, kernel_size = (3, 3), stride = 1, padding = 1)#512x512
        
        
    def forward(self, x):
        
        #ENCODER
        
        x1 = F.leaky_relu(self.convE1(x))
        x1_concat = F.leaky_relu(self.convE2(x1))
        
        x2 = F.max_pool2d(x1_concat, kernel_size = (2, 2), stride = 2, padding = 0)#256x256
        x2 = F.leaky_relu(self.convE3(x2))
        x2_concat = F.leaky_relu(self.convE4(x2))
        
        x3 = F.max_pool2d(x2_concat, kernel_size = (2, 2), stride = 2, padding = 0)#128x128
        x3 = F.leaky_relu(self.convE5(x3))
        x3_concat = F.leaky_relu(self.convE6(x3))
        
        x4 = F.max_pool2d(x3_concat, kernel_size = (2, 2), stride = 2, padding = 0)#64x64
        x4 = F.leaky_relu(self.convE7(x4))
        x4_concat = F.leaky_relu(self.convE8(x4))
        
        x5 = F.max_pool2d(x4_concat, kernel_size = (2, 2), stride = 2, padding = 0)#32x32
        x5 = F.leaky_relu(self.convE9(x5))
        x5 = F.leaky_relu(self.convE10(x5))
        
        #DECODER
        
        x5 = self.convT1(x5)
        x5 = torch.cat((x5, x4_concat), 1)
        x5 = F.leaky_relu(self.convD1(x5))
        x5 = F.leaky_relu(self.convD2(x5))
        
        x5 = self.convT2(x5)
        x5 = torch.cat((x5, x3_concat), 1)
        x5 = F.leaky_relu(self.convD3(x5))
        x5 = F.leaky_relu(self.convD4(x5))
        
        x5 = self.convT3(x5)
        x5 = torch.cat((x5, x2_concat), 1)
        x5 = F.leaky_relu(self.convD5(x5))
        x5 = F.leaky_relu(self.convD6(x5))
        
        x5 = self.convT4(x5)
        x5 = torch.cat((x5, x1_concat), 1)
        x5 = F.leaky_relu(self.convD7(x5))
        x5 = F.leaky_relu(self.convD8(x5))
        x5 = torch.sigmoid(self.convD9(x5))
        
        return x5

unet = UNet()
unet.cuda()