import torch
from PIL import Image
from rembg import remove as removefilter
import cv2
import torchvision.transforms as transform
import keyboard
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor as tensor
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=20, stride=20)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6, stride=6)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1280, 100),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softmax()
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, input):
        input = self.layer1(input)
        input = self.layer2(input)
        input = input.view((1, -1))
        input = self.layer3(input)
        input = self.layer4(input)
        return input


model = CNN()
model.load_state_dict(torch.load('regression_model.pth'))
model.eval()
totensor = transform.ToTensor()

while True:
    cap = cv2.VideoCapture(0)
    success, img = cap.read()



    if keyboard.is_pressed(' '):
        cv2.imwrite('now_image.png', img)
        image = Image.open('now_image.png')
        path='now_image.png'
        removefilter(image).save(path)
        cv2.imwrite(path, cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGBA2RGB))
        image = Image.open(path)
        image=totensor(image)
        predict=model.forward(image)
        print(predict)
        text = 'none'
        if predict[0][0] > 0.5:
            text = 'happy'
        if predict[0][1] > 0.5:
            text = 'sad'
        if predict[0][2] > 0.5:
            text = 'angry'
        print(text)
        print()

        cv2.putText(img, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(1)
    else:
        cv2.imshow('image', img)
        cv2.waitKey(1)