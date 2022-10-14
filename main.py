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

happynum =  8   +2  +
sadnum   =  8   +0
angrynum =  8   +0
sosonum  =  4   +2


totensor = transform.ToTensor()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    cv2.imshow('image', img)
    cv2.waitKey(1)

    if not success:
        print('카메라 못쓰는 듯')
        break

    if keyboard.is_pressed('h'):
        cv2.imwrite('now_image.png', img)
        img = Image.open('now_image.png')
        path=f'.\happy\img_happy{happynum}.png'
        removefilter(img).save(path)
        cv2.imwrite(path, cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGBA2RGB))

        happynum += 1
    elif keyboard.is_pressed('s'):
        cv2.imwrite('now_image.png', img)
        img = Image.open('now_image.png')
        path=f'.\sad\img_sad{sadnum}.png'
        removefilter(img).save(path)
        cv2.imwrite(path, cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGBA2RGB))

        sadnum += 1
    elif keyboard.is_pressed('a'):
        cv2.imwrite('now_image.png', img)
        img = Image.open('now_image.png')
        path=f'.\\angry\img_angry{angrynum}.png'
        removefilter(img).save(path)
        cv2.imwrite(path, cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGBA2RGB))

        angrynum += 1
    elif keyboard.is_pressed('n'):
        cv2.imwrite('now_image.png', img)
        img = Image.open('now_image.png')
        path=f'.\soso\img_soso{sosonum}.png'
        removefilter(img).save(path)
        cv2.imwrite(path, cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGBA2RGB))

        sosonum += 1
    elif keyboard.is_pressed(' '):
        break
print(happynum, sadnum, angrynum, sosonum)
cap.release()
