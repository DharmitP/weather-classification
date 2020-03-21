#from ImageCleaning import *

#using the trained neural network, evaluate its performance on a data set
def evaluate(net, loader, criterion=nn.BCEWithLogitsLoss(), testing=False, showImages=False):
    totalLoss = 0
    totalAcc = 0
    total0Acc = 0
    total1Acc = 0
    total0 = 0
    total1 = 0
    totalEpoch = 0

    for i, data in enumerate(loader, 0):
        #get the inputs
        inputs, labels = data

        if showImages:
            images = inputs.numpy() #convert images to numpy for display
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        #get the predictions
        outputs = net(inputs)

        #evaluate the loss and accuracy
        loss = criterion(outputs, labels.float())
        cacc = (outputs > 0).squeeze().long() == labels
        
        #evaluate true positives and true negatives
        cfalse = 0
        ctrue = 0
        if testing:
            for i, output in enumerate(outputs.squeeze(), 0):
                if labels[i].item() == 0:
                    total0 += 1
                    if output.item() <= 0:
                        cfalse += 1
                elif labels[i].item() == 1:
                    total1 += 1
                    if output.item() > 0:
                        ctrue += 1

        totalAcc += int(cacc.sum())
        totalLoss += loss.item()
        totalEpoch += len(labels)
        total0Acc += cfalse
        total1Acc += ctrue
        
        #print the images, their labels and whether the model was correct or not
        if showImages:
            classes = ["NotSnow", "Snow"]
            #plot the pictures, labels and if the model got it right
            fig = plt.figure(figsize=(25,  4 * (len(labels)//10+1)))
            for idx in range(len(labels)):
                ax = fig.add_subplot(len(labels)//10+1, 10, idx+1, xticks=[], yticks=[])
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                picture = np.transpose(images[idx], (1, 2, 0))
                picture = std * picture + mean
                picture = np.clip(picture, 0, 1)
                plt.imshow(picture)
                guess = int(outputs.squeeze()[idx].item() > 0)
                if guess == labels[idx].item():
                    isCorrect = ' O'
                else:
                    isCorrect = ' X'
                ax.set_title(classes[labels[idx].item()] + isCorrect)
    
    #calculate the final accuracy and loss
    acc = float(totalAcc) / totalEpoch
    loss = float(totalLoss) / (i + 1)
    if not testing:
        return acc, loss
    else:
        return acc, total0Acc / total0, total1Acc / total1

#example of usage
#IMPORTANT: if you saved ResNet50 features to Google Drive
#you have to load IMAGES instead of those features and feed it to
#resnet50, then to the rest of your model's layers

#Generate a name for the model consisting of all the hyperparameter values
def getModelName(size=320, batchSize=500, learningRate=0.0008, epoch=29):
    return "/content/gdrive/My Drive/Colab Notebooks/ProjectData/SnowData/Models/size{0}_bs{1}_lr{2}_epoch{3}".format(size, batchSize, learningRate, epoch)

import torchvision.models
resnet50 = torchvision.models.resnet50(pretrained=True) #load ResNet50

class SnowClassifier(nn.Module):
    def __init__(self, fcSize, getFeatures=False, useCuda=True):
        self.size = fcSize
        self.getFeatures = getFeatures
        self.useCuda = useCuda
        super(SnowClassifier, self).__init__()
        self.fc1 = nn.Linear(1000, self.size)
        self.fc2 = nn.Linear(self.size, self.size//2)
        self.fcf = nn.Linear(self.size//2, 1)

    def forward(self, x):
        if self.getFeatures:
            if self.useCuda:
                x = resnet50.cuda()(x)
            else:
                x = resnet50(x)
        x = x.view(-1, 1000)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fcf(x)
        return x.squeeze(1)
    
size = 750
batchSize = 30
learningRate = 0.0007
epoch = 49

model = SnowClassifier(size, True, True)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(getModelName(size, batchSize, learningRate, epoch)))

imagePath = "/content/gdrive/My Drive/Colab Notebooks/ProjectData/SnowData/"
testingImages = loadImages(imagePath + "Testing")
testLoader = torch.utils.data.DataLoader(testingImages, batch_size=batchSize, num_workers=1, shuffle=True)

acc, acc0, acc1 = evaluate(model, testLoader, nn.BCEWithLogitsLoss(), True, True)
print(acc)
print(acc0)
print(acc1)
