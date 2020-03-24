import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

train_data = datasets.CIFAR10('CIFAR10', train = True, 
                         transform = transforms.Compose([transforms.ToTensor()]),
                         download = True)

test_data = datasets.CIFAR10('CIFAR10', train = False,
                             transform = transforms.Compose([transforms.ToTensor()]),
                             download = True)

train_batch = torch.utils.data.DataLoader(train_data, batch_size = 100, shuffle = True)
test_batch = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size = (3, 3), stride = 1, padding = 1) #  32x32
        '''In the original paper the authors have drastically reduced the resolution of the image but since the image size here 
        is already very small therefore preserved the dimension in the first convolution operation'''
        self.conv2 = nn.Conv2d(96, 256, kernel_size = (3, 3), stride = 1, padding = 1) # 16x16
        self.conv3 = nn.Conv2d(256, 384, kernel_size = (3, 3), stride = 1, padding = 1) #8x8
        self.conv4 = nn.Conv2d(384, 384, kernel_size = (3, 3), stride = 1, padding = 1) #8x8
        self.conv5 = nn.Conv2d(384, 256, kernel_size = (3, 3), stride = 1, padding = 1) #8x8
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)
        
    def convs(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = (2, 2), stride = 2, padding = 0) # 16x16
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = (2, 2), stride = 2, padding = 0) # 8x8
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = (2, 2), stride = 2, padding = 0) #4x4
        return x
        
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim = 1)
        return x
    
alexnet = AlexNet()

optimizer = optim.Adam(alexnet.parameters(), lr = 0.001)
loss_function = nn.BCELoss()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Running on GPU")
else:
    device = torch.device('cpu')
    print("Running on cpu")
    
alexnet.to(device)

def one_hot_encoder(labels, num_labels):
    batch_size = len(labels)
    one_hot_labels = np.zeros([batch_size, num_labels])
    for i in range(batch_size):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels

EPOCHS = 1

def train(train_batch):
    for epoch in range(EPOCHS):
        for images, labels in tqdm(train_batch):
            one_hot_labels = torch.Tensor(one_hot_encoder(labels, 10))
            alexnet.zero_grad()
            outputs = alexnet.forward(images.to(device))
            loss = loss_function(outputs, one_hot.to(device))
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}. Loss: {loss}")
        

def test(test_batch):
    correct = 0
    total = 0
    for images, labels in tqdm(test_batch):
        labels = labels.to(device)
        net_out = alexnet(images.to(device))
        predicted_class = torch.argmax(net_out).to(device)
        if (predicted_class == labels):
            correct += 1
        total += 1
    print("Accuracy: ", round(correct/total, 3))
    
    
train(train_batch)
test(test_batch)