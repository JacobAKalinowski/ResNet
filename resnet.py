import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from fastai.layers import Flatten

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, dropout):
        super(ResidualBlock, self).__init__()
        #First Convolution
        self.convolution1 = nn.Conv2d(
            in_channels = in_channel,
            out_channels = out_channel,
            kernel_size = 3,
            padding = 1,
            stride = stride,
            bias = False
        )
        #Second Convolution
        self.convolution2 = nn.Conv2d(
            in_channels = out_channel,
            out_channels = out_channel,
            kernel_size = 3,
            padding = 1,
            stride = 1,
            bias = False
        )
        #Norms and dropout
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.dropout = nn.Dropout(dropout, inplace = False)
        self.relu = nn.ReLU()
        
        if in_channel == out_channel:
            self.downsample = lambda x: x #Downsample won't change x after 1st layer of the block
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channel,
                    out_channels = out_channel,
                    kernel_size = 3,
                    padding = 1,
                    stride = stride,
                    bias = False
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
        
    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.norm1(self.convolution1(x)))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.convolution2(out)))
        out = out+residual
        return out

class ResNet(nn.Module):
    def __init__(self, num_layers, layer_size = None, k=1, dropout = 0.3, width = 32):
        super(ResNet, self).__init__()
        #Initial convolution
        self.layers = [nn.Conv2d(
            in_channels = 3,
            out_channels = width,
            kernel_size = 3,
            padding = 1,
            bias = False
        )]
        
        widths = [width]
        
        for i in range(num_layers):
            widths.append(width*(2**i)*k)
        
        #Creating each new layer
        for i in range(num_layers):
            self.layers += self.new_layer(
                layer_size = layer_size,
                in_channel = widths[i],
                out_channel = widths[i+1],
                stride = 1 if i == 0 else 2,
                dropout = dropout
            )
        
        self.layers += [
            nn.BatchNorm2d(widths[-1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(widths[-1], 10)
        ]
        
        self.network = nn.Sequential(*self.layers)
        
    def new_layer(self, layer_size, in_channel, out_channel, stride, dropout):
        #Method to create each residual layer
        layer = []
        for i in range(layer_size):
            block = ResidualBlock(
                in_channel = (in_channel if i == 0 else out_channel),
                out_channel = out_channel,
                stride = stride if i == 0 else 1,
                dropout = dropout
            )
            layer.append(block)
        return layer
    
    def forward(self, x):
        return self.network(x)

#Create Transformation of the imageset into tensors in range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Load training and test data from CIFAR10 dataset
train_data_set = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transform
)

test_data_set = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,
    download = True,
    transform = transform
)

#Set batch size for iterator
batch_size = 32

#Create DataLoader for iterating
train_loader = torch.utils.data.DataLoader(
    train_data_set,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 2
)

test_loader = torch.utils.data.DataLoader(
    test_data_set,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 2
)

#Label the classes of dataset
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Create our ResNet:
model = ResNet(num_layers = 4, layer_size = 4, k=4)

#Set up optimizer and criterion
criterion = nn.CrossEntropyLoss()
lr = 0.001
momentum = 0.9
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
epochs = 50

#Send model to CUDA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

#Check the device we are using
print(device)

#Load Checkpoint if not training model from scratch
PATH = "resnet4X4.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

#Create Training loop
torch.autograd.set_detect_anomaly(True)
PATH = "resnet4x4.pt"

for epoch in range(epoch, epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #Get inputs and labels and send to the device
        inputs, labels = data[0].to(device), data[1].to(device)
        
        #Set parameter gradients to 0
        optimizer.zero_grad()
        
        #Forward
        outputs = model(inputs)
        
        #Loss
        loss = criterion(outputs, labels)
        
        #Backward
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        #Print running loss at every 100 batches
        if i % 100 == 99:    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    torch.save({
        'epoch' : epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, PATH)
    
    #Save at 1, 10, and 25 epochs for TSNE
    if(epoch == 1):
        torch.save({
        'epoch' : 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, "resnet4X4epoch1.pt")
        
    if(epoch == 10):
        torch.save({
        'epoch' : 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, "resnet4X4epoch10.pt")
        
    if(epoch == 25):
        torch.save({
        'epoch' : 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, "resnet4X4epoch25.pt")
        
print('Training Done')

#Check Accuracy
correct = 0
total = 0
#Don't compute gradient as we are done training
with torch.no_grad():
    for data in test_loader:
        #Get test data
        images, labels = data[0].to(device), data[1].to(device)
        #Pass test data through the model
        outputs = model(images)
        #Make prediction and add to total
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#Get accuracies for each class and store in dictionary
correct_prediction = {class_name: 0 for class_name in classes}
total_prediction = {class_name: 0 for class_name in classes}
accuracies = {class_name: 0 for class_name in classes}

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _,predictions = torch.max(outputs, 1)
        
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_prediction[classes[label]] += 1
            total_prediction[classes[label]] += 1
            
for class_name, correct_count in correct_prediction.items():
    accuracy = 100 * float(correct_count) / total_prediction[class_name]
    accuracies[class_name] = accuracy
    print(f'Accuracy for class: {class_name:5s} is {accuracy:.1f} %')


#Method to return representation vectors for TSNE visualization
def return_representations(resnet, data, device = "cpu", batch_size = 32):
    resnet.layers[-1] = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()
    
    representations = torch.Tensor([]).to(device)
    loader = torch.utils.data.DataLoader(data, batch_size = batch_size)
    for data, _ in loader:
        data = data.to(device)
        out = resnet(data).detach()
        representations = torch.concat([representations,out], dim = 0)
        
    return representations

from torch.utils.data import Subset
from random import sample
from sklearn.manifold import TSNE
import matplotlib

#Take subset of sample data
dataset = Subset(test_data_set, sample(range(len(test_data_set)), 5000))

labels = [dataset[i][1] for i in range(len(dataset))]
outs = []

for epoch in [1, 10, 25, 50]:
    print(epoch)
    PATH = f"resnet4X4epoch{epoch}.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    representations = return_representations(model, dataset, device = "cuda:0", batch_size = 32)
    
    tsne = TSNE(n_components = 2, n_iter = 1000)
    outs.append(tsne.fit_transform(representations.cpu()))

plt.figure(figsize=(12,10))
plt.subplot(2,2,1)

plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)

plt.suptitle("TSNE Visualization of CIFAR10 in Representation Space", fontsize=16)
plt.scatter(outs[0][:, 0], outs[0][:, 1], 1, labels)
plt.xlabel("1 Epoch", fontsize=12)
plt.gcf().patch.set_alpha(1)
plt.subplot(2,2,2)

plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
plt.scatter(outs[1][:, 0], outs[1][:, 1], 1, labels)
plt.xlabel("10 Epochs", fontsize=12)

plt.subplot(2,2,3)
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
plt.scatter(outs[2][:, 0], outs[2][:, 1], 1, labels)
plt.xlabel("25 Epochs", fontsize=12)

plt.subplot(2,2,4)
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
plt.scatter(outs[3][:, 0], outs[3][:, 1], 1, labels)
plt.xlabel("50 Epochs", fontsize=12)

#Now plot a bar graph for accuracies

keys = list(accuracies.keys())
values = list(accuracies.values())

fig =plt.figure()
plt.bar(keys, values)
for i in range(len(keys)):
    plt.text(i - 0.5, values[i]+1, values[i])

plt.xlabel("Class Name")
plt.ylabel("Accuracy on Class")
plt.title("Accuracy of ResNet Model on CIFAR10 Dataset")
plt.show()

