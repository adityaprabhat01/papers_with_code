import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

train = datasets.MNIST('', train = True, download = True,
                       transform = transforms.Compose([transforms.ToTensor()]))
train_set = torch.utils.data.DataLoader(train, batch_size = 100, shuffle = True)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Running on GPU")
else:
    device = torch.device('cpu')
    print("Running on cpu")



def latent_space_vectors(size): #size is the number of samples in a batch
    return torch.randn(size, 100).to(device)

def real_data_target(size):
    return (torch.ones(size, 1)).to(device)

def fake_data_target(size):
    return (torch.zeros(size, 1)).to(device)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_features = 100
        output_features = 784
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.out = nn.Linear(1024, output_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        input_features = 784
        output_features = 1
        self.fc1 = nn.Linear(input_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


loss_function = torch.nn.BCELoss()
generator = Generator()
discriminator = Discriminator()

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    loss_function.cuda()
    print(generator)
    print(discriminator)
    print(loss_function)


optimizer_generator = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002)


def train_discriminator(real_image, fake_image):
    optimizer_discriminator.zero_grad()
    
    #training discriminator using real images
    prediction_real_image = discriminator.forward(real_image)
    loss_real_image = loss_function(prediction_real_image, real_data_target(real_image.size(0)))
    loss_real_image.backward()
    
    #training discriminator using fake images
    prediction_fake_image = discriminator.forward(fake_image)
    loss_fake_image = loss_function(prediction_fake_image, fake_data_target(prediction_fake_image.size(0)))
    loss_fake_image.backward()
    
    optimizer_discriminator.step()
    
    return loss_fake_image + loss_real_image, prediction_real_image, prediction_fake_image


def tarin_generator(fake_image):
    optimizer_generator.zero_grad()
    prediction_fake_image = discriminator.forward(fake_image)
    error_fake_image = loss_function(prediction_fake_image, real_data_target(prediction_fake_image.size(0)))
    error_fake_image.backward()
    optimizer_generator.step()
    
    return error_fake_image


EPOCHS = 10

for epoch in range(EPOCHS):
    for real_batch,_ in tqdm(train_set):

        real_image = real_batch.view(real_batch.size(0), 784).to(device)
        fake_image = generator.forward(latent_space_vectors(100))
        d_error, d_pred_real, d_pred_fake = train_discriminator(real_image, fake_image)
        
        
        fake_image = generator.forward(latent_space_vectors(100))
        g_error = tarin_generator(fake_image)
        
    print("Discriminator loss = " + str(d_error.data.cpu().numpy()) + " Generator loss = " + str(g_error.data.cpu().numpy()))