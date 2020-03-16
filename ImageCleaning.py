import torch
import numpy as np
import time
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

#mount googledrive
from google.colab import drive
drive.mount('/content/gdrive')

#resize and black pad images to 224x224 with no distortion
def cleaner(file):
    #resize image while keeping aspect ratio (e.g. 500x375 becomes 224x168)
    x = 224
    y = 224
    image = Image.open(file)
    image.thumbnail((x, y), Image.ANTIALIAS)

    #add symmetrical black padding to the smaller dimension to match the desired x and y
    deltaX = x - image.size[0]
    deltaY = y - image.size[1]
    padding = (deltaX//2, deltaY//2, deltaX-(deltaX//2), deltaY-(deltaY//2))

    return ImageOps.expand(image, padding)

#load gesture images from Google Drive
def loadImages(folder):
    startTime = time.time() 

    #transform to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    #load data from Google Drive and clean them as they load
    dataset = torchvision.datasets.ImageFolder(root=folder, loader=cleaner, transform=transform)
    
    return dataset

#verify that the images have been loaded and labeled
def verifyImages(dataset):
    #prepare dataloader
    startTime = time.time()
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=1, shuffle=True)
    startTime = time.time()

    #verification step - obtain one batch of images
    dataiter = iter(dataLoader)
    images, labels = dataiter.next()
    images = images.numpy() #convert images to numpy for display
    classes = ["Not Snow", "Snow"]

    #plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(6):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(classes[labels[idx]])

#example usage with folder named "Overfit"
#within the folder "Overfit" there are two subfolders named "NotSnow" and "Snow"
#the subfolder name which is alphabetically first will be labelled as class 0
#the second folder will be labelled as class 1
overfitImages = loadImages("/content/gdrive/My Drive/Colab Notebooks/ProjectData/Overfit")
verifyImages(overfitImages)
