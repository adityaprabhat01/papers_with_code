import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, utils

train_data = datasets.CIFAR10('CIFAR10', train = True, 
                         transform = transforms.Compose([transforms.ToTensor()]),
                         download = True)

train_batch = torch.utils.data.DataLoader(train_data, batch_size = 100, shuffle = True)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Running on cpu")
    
    
def latent_space_vectors(size): #size is the number of samples in a batch
    return torch.randn(size, 100, 1, 1).to(device)

def real_data_target(size):
    return (torch.ones(size, 1)).to(device)

def fake_data_target(size):
    return (torch.zeros(size, 1)).to(device)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convT1 = nn.ConvTranspose2d(100, 256, kernel_size = 4, stride = 1, padding = 0)
        self.batch_norm1 = nn.BatchNorm2d(256)
        
        self.convT1_labels = nn.ConvTranspose2d(10, 256, kernel_size = 4, stride = 1, padding = 0)
        self.batch_norm1_labels = nn.BatchNorm2d(256)
        
        self.convT2 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.convT3 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.convT4 = nn.ConvTranspose2d(128, 3, kernel_size = 4, stride = 2, padding = 1)
        
    def forward(self, x, y):
        x = self.convT1(x)
        x = F.relu(self.batch_norm1(x))
        y = self.convT1_labels(y)
        y = F.relu(self.batch_norm1_labels(y))
        x = torch.cat((x,y), 1)
        x = self.convT2(x)
        x = F.relu(self.batch_norm2(x))
        x = self.convT3(x)
        x = F.relu(self.batch_norm3(x))
        x = self.convT4(x)
        x = torch.tanh(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_label = nn.Linear(10, 1024)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 4, stride = 2, padding = 1)
        
        self.conv1_labels = nn.Conv2d(1, 32, kernel_size = 4, stride = 2, padding = 1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size = 2, stride = 1, padding = 0) #4,1,0
        
    def forward(self, x, y):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        y = self.linear_label(y)
        y = y.view(100, 1, 32, 32)
        y = self.conv1_labels(y)
        y = F.leaky_relu(y, 0.2)
        x = torch.cat((x, y), 1)
        x = self.conv2(x)
        x = F.leaky_relu(self.batch_norm1(x), 0.2)
        x = self.conv3(x)
        x = F.leaky_relu(self.batch_norm2(x), 0.2)
        x = self.conv4(x)
        x = F.leaky_relu(self.batch_norm3(x), 0.2)
        x = self.conv5(x)
        x = torch.sigmoid(x)
        return x
    
generator = Generator()
generator.to(device)
discriminator = Discriminator()
discriminator.to(device)

loss_function = torch.nn.BCELoss()
optimizer_generator = optim.Adam(generator.parameters(), lr = 0.0002)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = 0.0002)


def one_hot_encoder(labels, num_labels):
    batch_size = len(labels)
    one_hot_labels = np.zeros([batch_size, num_labels])
    for i in range(batch_size):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels


def train_generator(fake_image, fake_image_labels):
    optimizer_generator.zero_grad()
    
    prediction_fake_image = discriminator.forward(fake_image, fake_image_labels)
    loss_fake_image = loss_function(prediction_fake_image, real_data_target(prediction_fake_image.size(0)))
    loss_fake_image.backward()
    optimizer_generator.step()
    
    return loss_fake_image


def train_discriminator(real_image, fake_image, real_labels, fake_labels):
    optimizer_discriminator.zero_grad()
    
    #train on real images
    prediction_real_image = discriminator.forward(real_image, real_labels)
    loss_real_image = loss_function(prediction_real_image, real_data_target(prediction_real_image.size(0)))
    loss_real_image.backward()
    
    #train on fake images
    prediction_fake_image = discriminator.forward(fake_image, fake_labels)
    loss_fake_image = loss_function(prediction_fake_image, fake_data_target(prediction_fake_image.size(0)))
    loss_fake_image.backward()
    
    optimizer_discriminator.step()
    
    return loss_real_image + loss_fake_image, prediction_real_image, prediction_fake_image


EPOCHS = 1

for i in range(EPOCHS):
    for real_batch, labels in tqdm(train_batch):
        
        real_image = real_batch.to(device)
        real_labels = torch.Tensor(one_hot_encoder(labels, 10)).to(device)
        fake_labels = torch.Tensor(torch.randn(100, 10, 1, 1)).to(device)
        fake_image = generator.forward(latent_space_vectors(100), fake_labels)
        d_error, d_pred_real, d_pred_fake = train_discriminator(real_image, fake_image, real_labels, fake_labels.view(100, 10))
        
        fake_labels = torch.Tensor(torch.randn(100, 10, 1, 1)).to(device)
        fake_image = generator.forward(latent_space_vectors(100), fake_labels).to(device)
        g_error = train_generator(fake_image, fake_labels.view(100, 10))

    print("Discriminator loss = " + str(d_error.data.cpu().numpy()) + " Generator loss = " + str(g_error.data.cpu().numpy()))
    
