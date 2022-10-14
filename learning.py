import torch
from PIL import Image
from rembg import remove as removㅜefilter
import cv2
import torchvision.transforms as transform
import keyboard
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor as tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

#gpu를 사용해야 할 것 같다
print(torch.cuda.is_available())



happynum = 8
sadnum   = 8
angrynum = 8
sosonum  = 4

batch_size=7
epoches=400

totensor = transform.ToTensor()

x_train=[]
y_train=[]
for i in range(happynum):
    image = Image.open(f'.\happy\img_happy{i}.png')
    image=totensor(image).detach().numpy()
    x_train.append(image)
    y_train.append([1, 0, 0])
for i in range(sadnum):
    image = Image.open(f'.\sad\img_sad{i}.png')
    image=totensor(image).detach().numpy()
    x_train.append(image)
    y_train.append([0, 1, 0])
for i in range(angrynum):
    image = Image.open(f'.\\angry\img_angry{i}.png')
    image=totensor(image).detach().numpy()
    x_train.append(image)
    y_train.append([0, 0, 1])
for i in range(sosonum):
    image = Image.open(f'.\\soso\img_soso{i}.png')
    image=totensor(image).detach().numpy()
    x_train.append(image)
    y_train.append([1/3, 1/3, 1/3])

x_train=tensor(x_train)
y_train=tensor(y_train)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.initalmaxpool=nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=21, padding=10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=20, stride=20)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6, stride=6)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1280, 50),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(50, 3),
            nn.Softmax()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        #input=self.initalmaxpool(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = input.view((batch_size, -1))
        input = self.layer3(input)
        input = self.layer4(input)
        return input

model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.07)

datasets = TensorDataset(x_train, y_train)
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

for epoch in range(epoches):
    cost_sum=0
    i=0
    for batch_ind, sample in enumerate(dataloader):
        x, y = sample
        prediction = model(x)
        cost = F.cross_entropy(prediction, y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        cost_sum += cost.item()
        i+=1
    print("epoch {} : {:.5f}".format(epoch+1, cost_sum/i))

torch.save(model.state_dict(), 'regression_model.pth')
