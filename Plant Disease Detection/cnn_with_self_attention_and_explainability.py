# -*- coding: utf-8 -*-
"""CNN with self-attention and explainability.ipynb

"""#### Import Libraries"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from einops import rearrange, repeat
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.cuda.amp as amp

"""#### Attach Dataset"""

from google.colab import drive
drive.mount('/content/drive')

"""#### Define CNN Architecture with self-attention mechanism

"""

# Define the CNN class with attention mechanism
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.LogSoftmax(dim=1)

        # Define the attention mechanism
        self.attn = MultiheadAttention(embed_dim=512, num_heads=1)

    def forward(self, x):
        intermediate_feature_maps = []

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        intermediate_feature_maps.append(x.clone().detach())

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        intermediate_feature_maps.append(x.clone().detach())

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        intermediate_feature_maps.append(x.clone().detach())  # Save the feature maps before self-attention

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)

        # Apply self-attention mechanism
        attention_output, _ = self.attn(x, x, x)
        # Process attention_output to obtain attention scores or maps

        # Combine attention scores or maps with feature maps
        combined_feature_maps = attention_output.unsqueeze(-1).unsqueeze(-1) * x.unsqueeze(-1).unsqueeze(-1)

        x = self.fc2(x)
        intermediate_feature_maps.append(x.clone().detach())  # Save the feature maps after self-attention
        output = self.softmax(x)

        return output, intermediate_feature_maps, combined_feature_maps

"""#### GradCAM Implementation"""

# Function to visualize images and CAMs
def visualize_cam(image_tensor, grayscale_cam):
    img = image_tensor.cpu().numpy().transpose((1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title('Grad-CAM')
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define transforms for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define path to the root folder of your dataset
dataset_root_folder = '/content/drive/My Drive/PlantVillage'

# Create datasets
full_dataset = ImageFolder(dataset_root_folder, transform=transform)

# Print the number of classes and list all the classes
print(f"Number of classes: {len(full_dataset.classes)}")
print("Classes:")
for class_name in full_dataset.classes:
    print(class_name)
print("\n")

# Determine the number of samples for the reduced dataset
reduced_dataset_size = int(len(full_dataset) * 0.01)

# Randomly sample 5% of the dataset
reduced_dataset, _ = random_split(full_dataset, [reduced_dataset_size, len(full_dataset) - reduced_dataset_size])

# Split the reduced dataset into training and testing sets
train_size = int(0.8 * len(reduced_dataset))
test_size = len(reduced_dataset) - train_size
train_dataset, test_dataset = random_split(reduced_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Check if CUDA is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Setup Grad-CAM
target_layers = [model.conv3]
cam = GradCAM(model=model, target_layers=target_layers)

# Train the model
num_epochs = 10
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, intermediate_feature_maps, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Store average training loss for this epoch
    train_losses.append(train_loss / len(train_loader))
    print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs, intermediate_feature_maps, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Generate and visualize CAMs for a subset of images
        if epoch == num_epochs - 1:  # Only visualize CAMs during the last epoch
            for idx in range(images.size(0)):
                input_tensor = images[idx].unsqueeze(0)
                targets = [ClassifierOutputTarget(predicted[idx].item())]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                visualize_cam(images[idx], grayscale_cam)  # Generates the CAM images in the last epoch

    # Calculate and store accuracy for this epoch
    accuracy = accuracy_score(all_labels, all_predictions)
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}, Test Accuracy: {accuracy * 100:.2f}%")

# Calculate and print F1 score for the last epoch
f1 = f1_score(all_labels, all_predictions, average='weighted')
print(f"F1 Score on test set: {f1:.2f}")

# Plot the loss and accuracy graphs
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, 'b', label='Test accuracy')
plt.title('Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

