import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, datasets

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



#for celebrity face dataset uncomment the commented lines

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convT1 = nn.ConvTranspose2d(100, 512, kernel_size = 4, stride = 1, padding = 0)
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.convT2 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.convT3 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.convT4 = nn.ConvTranspose2d(128, 3, kernel_size = 4, stride = 2, padding = 1) #change 3 to 64
        #self.batch_norm4 = nn.BatchNorm2d(64)
        #self.convT5 = nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1)
        
    def forward(self, x):
        x = self.convT1(x)
        x = F.relu(self.batch_norm1(x))
        x = self.convT2(x)
        x = F.relu(self.batch_norm2(x))
        x = self.convT3(x)
        x = F.relu(self.batch_norm3(x))
        x = self.convT4(x)
        #x = F.relu(self.batch_norm4(x))
        #x = self.convT5(x)
        x = torch.tanh(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size = 2, stride = 1, padding = 0) #4,1,0
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(self.batch_norm1(x), 0.2)
        x = self.conv3(x)
        x = F.leaky_relu(self.batch_norm2(x), 0.2)
        x = self.conv4(x)
        x = F.leaky_relu(self.batch_norm3(x), 0.2)
        x = self.conv5(x)
        x = torch.sigmoid(x)
        return x



def latent_space_vectors(size): #size is the number of samples in a batch
    return torch.randn(size, 100, 1, 1).to(device)

def real_data_target(size):
    return (torch.ones(size, 1)).to(device)

def fake_data_target(size):
    return (torch.zeros(size, 1)).to(device)


discriminator = Discriminator()
generator = Generator()
loss_function = torch.nn.BCELoss()
discriminator.to(device)
generator.to(device)
optimizer_generator = optim.Adam(generator.parameters(), lr = 0.0002)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = 0.0002)


def train_discriminator(real_image, fake_image):
    optimizer_discriminator.zero_grad()
    
    #training discriminator using real images
    prediction_real = discriminator.forward(real_image)
    loss_real_image = loss_function(prediction_real, real_data_target(prediction_real.size(0)))
    loss_real_image.backward()
    
    #training discriminator using fake image
    prediction_fake = discriminator.forward(fake_image)
    loss_fake_image = loss_function(prediction_fake, fake_data_target(prediction_fake.size(0)))
    loss_fake_image.backward()
    
    optimizer_discriminator.step()
    
    return loss_real_image + loss_fake_image, prediction_real, prediction_fake


def train_generator(fake_image):
    optimizer_generator.zero_grad()
    
    prediction_fake_image = discriminator.forward(fake_image)
    loss_fake_image = loss_function(prediction_fake_image, real_data_target(prediction_fake_image.size(0)))
    loss_fake_image.backward()
    optimizer_generator.step()
    
    return loss_fake_image


EPOCHS = 1

for epochs in range(EPOCHS):
    for real_batch, _ in tqdm(train_batch):
        
        real_image = real_batch.to(device)
        fake_image = generator.forward(latent_space_vectors(100))
        d_error, d_pred_real, d_pred_fake = train_discriminator(real_image, fake_image)
        
        fake_image = generator.forward(latent_space_vectors(100))
        g_error = train_generator(fake_image)
        
    print("Discriminator loss = " + str(d_error.data.cpu().numpy()) + " Generator loss = " + str(g_error.data.cpu().numpy()))
