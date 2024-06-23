# %%
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
class CNN(nn.Module):
    def __init__(self,image_size, channels, num_classes):
        super(CNN, self).__init__() # Calling the parent constructor

        # initialize first set of CONV => RELU => POOL layers
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1) 
        im_size = list(image_size)
        op = ((im_size[0] - 3 + 2*(1))/1) + 1
        #print(op)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        op = op/2
        #print(op)
        
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1)
        op = ((op - 3 + 2*(1))/1) + 1
        #print(op)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride = 2)
        op = op/2
        #print(op)

        # initialize third set of CONV => RELU => POOL layers
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1)
        op = int(((op - 3 + 2*(1))/1) + 1)
        #print(op)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.dropout = nn.Dropout(0.25)

        # initialize FC layers
        self.fc1 = nn.Linear(channels[3]*op*op,channels[4])
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(channels[4],num_classes)
        self.softmax = nn.LogSoftmax(dim=1)                     
        #self.dropout = nn.Dropout(0.2)
    
    # Basically defines the order in which the layers will be applied
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        output = self.softmax(x)
        
        return output

# %%
# Defining various architectures

def cnn_1():           
    num_classes = 60
    channels = [1,32,64,128,256]
    return CNN((28,28),channels, num_classes)

def cnn_2():           
    num_classes = 10
    channels = [1,16,32,64,128]
    return CNN((28,28),channels, num_classes)

# %%
num_epochs = 500
batchsize_train = 64
batchsize_test = 1000
learning_rate = 0.03
beta1 = 0.9
beta2 = 0.99

# %%
# To transform images 
transform_func = transforms.Compose([ 
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,),(0.3081,))
])

# %%
# Loading the training dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/prachikansal/Desktop/sem6/assignments/dl/assignment 2', train=True, download=True, 
    transform=transform_func), 
    batch_size=batchsize_train, shuffle=True
)

# Loading the test dataset
test_loader= torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/prachikansal/Desktop/sem6/assignments/dl/assignment 2', train=False, download=True,
    transform=transform_func), 
    batch_size=batchsize_test, shuffle=True
)

# %%
# Getting the datasets
train_dataset = train_loader.dataset
test_dataset = test_loader.dataset

# Classes in the MNIST dataset
print(f"Number of classes: {len(train_dataset.classes)}")
print("Classes: ")
for class_name in train_dataset.classes:
    print(class_name)
print("\n")

# %%
# Displaying some samples from the training dataset
for images, _ in train_loader:
    print('images.shape: ',images.shape)
    plt.figure(figsize = (16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1,2,0)))
    break

# %%
# To count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
# Defining the model
model = cnn_2()
print(f"Model Parameters: {count_parameters(model)}")

# Defining the loss function
loss_func = nn.CrossEntropyLoss()

# Defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(beta1, beta2), eps = 1e-08, weight_decay=1e-03)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.88) # Exponentially reduces the learning rate with each epoch

# %% [markdown]
# - Train batch size = 64
# - Number of images in train set = 60,000
# - Number of batches = ceil(60000/64) = 938
# 

# %%
# Defining an array to store accuracies on training and test set for different epochs
train_accuracy_list = []
test_accuracy_list = []

# Train the model
for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    train_loss = 0.0 # Initially the training loss is zero
    for images, labels in tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    train_accuracy_list.append(train_accuracy)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Train Accuracy: {100 * correct / total:.5f}%")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc="Testing"):
        outputs = model(images) # Getting the predictions 
        _, predicted = torch.max(outputs.data, 1) # The predicted variable stores these predicted classes. 
        # The _ is used to ignore the actual highest probabilities, as we're only interested in the predicted classes

        for i in range(len(images)):
            total +=1
            if (predicted[i] == labels[i]):
                correct +=1

    test_accuracy = correct / total
    test_accuracy_list.append(test_accuracy)    
    print(f"Epoch {epoch+1}, Accuracy on test set: {100 * correct / total:.5f}%")
    print("\n")

    scheduler.step()

# %%
# Excel sheet to store accuracies for each epoch  
columns = ['Epoch', 'Train Accuracy', 'Test Accuracy']

data = {'Epoch': list(range(1, 501)),
        'Train Accuracy': train_accuracy_list,
        'Test Accuracy': test_accuracy_list}

df = pd.DataFrame(data, columns=columns)
df.to_excel("MNIST_Accuracy_Results.xlsx", index=False)


