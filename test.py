import torch
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from __init__ import Net, imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Define the image loader function
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32,32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_image(path):
    image = Image.open(path)
    x = transform(image)
    x = x.unsqueeze(0)
    return x

#Load and prepare test dataset
batch_size = 1





testset = datasets.ImageFolder('./Dataset/test', transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


classes = ('anger', 'disgust', 'fear', 'happiness',
           'neutral', 'sadness', 'surprise')
#Load and prepare nn

net = Net()
net.load_state_dict(torch.load("./expression.pth"))
net.eval()

im = load_image('./individual_test/sad.jpg')
outputs = net(im)
b = outputs.data
pred = torch.argmax(outputs, 1)
print(classes[pred])


#Test accuracy of nn


