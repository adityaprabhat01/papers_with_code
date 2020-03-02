import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

training_data = np.load('training_data.npy', allow_pickle = True)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size = 11, stride = 4, padding = 0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)
        
    def convs(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = (3, 3), stride = 2, padding = 0)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = (3, 3), stride = 2, padding = 0)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = (3, 3), stride = 2, padding = 0)
        return x
        
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim = 1)
        return x
    
alexnet = AlexNet()

optimizer = optim.Adam(alexnet.parameters(), lr = 0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 227, 227)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
del training_data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
del X

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Running on GPU")
else:
    device = torch.device('cpu')
    print("Running on cpu")

alexnet.to(device)

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
        X_batch = X_train[i:i+BATCH_SIZE].view(-1,1,227,227).to(device)
        y_batch = y_train[i:i+BATCH_SIZE].to(device)
        alexnet.zero_grad()
        outputs = alexnet(X_batch)
        loss = loss_function(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}. Loss: {loss}")

def test(alexnet):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            real_class = torch.argmax(y_test[i]).to(device)
            net_out = alexnet(X_test[i].view(-1, 1, 227, 227).to(device))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct/total, 3))

test(alexnet)