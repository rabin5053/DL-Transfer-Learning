# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
The problem statement for this experiment is to develop an image classification model that can accurately distinguish between 'defect' and 'notdefect' semiconductor chip images. This is a binary classification task, where the goal is to leverage transfer learning using a pre-trained VGG19 model to effectively classify new, unseen chip images.

## Neural Network Model

<img width="1043" height="802" alt="592394652-2f989398-134b-4cef-8fb1-6efb6aed0055" src="https://github.com/user-attachments/assets/c67d64fe-6d90-46b2-84b2-84a9bf1d77e0" />


## DESIGN STEPS

### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.



## PROGRAM

### Name: Rabin R

### Register Number: 212224230213

```python
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])

!unzip -qq /content/drive/MyDrive/deeplearningexperiments/chip_data.zip -d data

from torchvision import datasets
import torchvision.transforms as transforms

# Define transformations for images (moved from oymch2pTUreT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])

# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()


# Show sample images from the training dataset
show_sample_images(train_dataset)

# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Get the total number of samples in the testing dataset
print(f'Total number of test samples: {len(test_dataset)}')


# Get the shape of the first image in the dataset
first_image1,label=test_dataset[0]
print(f"Shape of the first image: {first_image1.shape}")

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## Step 2: Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
model=models.vgg19(weights=models.VGG19_Weights.DEFAULT)


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train() 
    for epoch in range(num_epochs):
      running_loss=0.0
      for images, labels in train_loader: 
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images) 
        loss=criterion(outputs,labels.unsqueeze(1).float()) 
        loss.backward() 
        optimizer.step()
        running_loss+=loss.item()
      train_losses.append(running_loss/len(train_loader)) 

      # Compute validation loss
      model.eval()
      val_loss=0.0
      with torch.no_grad():
        for images,labels in test_loader: 
          images,labels=images.to(device),labels.to(device)
          outputs=model(images) 
          loss=criterion(outputs,labels.unsqueeze(1).float())
          val_loss+=loss.item()
      val_losses.append(val_loss/len(test_loader))
      model.train()

      
      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss 
    print("Name: Thrinesh Royal")
    print("Register Number: 212223230226")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the model
train_model(model,train_loader,test_loader)

## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs=torch.sigmoid(outputs)
            predicted=(probs>0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Name: Thrinesh Royal")
    print("Register Number: 212223230226")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Name: Thrinesh Royal")
    print("Register Number: 212223230226")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

test_model(model,test_loader)

## Step 5: Predict on a Single Image and Display It
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Apply sigmoid to get probability, threshold at 0.5
        prob = torch.sigmoid(output)
        predicted = (prob > 0.5).int().item()


    class_names = class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)


predict_image(model,image_index=25,dataset=test_dataset)


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="1344" height="645" alt="image" src="https://github.com/user-attachments/assets/3f0d1b60-6df4-4d5d-b004-80432abfa445" />


<img width="865" height="692" alt="image" src="https://github.com/user-attachments/assets/ef409280-1d53-46b6-b7ea-4bfe8be679c8" />



## Confusion Matrix
<img width="338" height="315" alt="image" src="https://github.com/user-attachments/assets/0d8227b0-b9c0-41ca-bb38-061895772465" />





## Classification Report
<img width="530" height="311" alt="image" src="https://github.com/user-attachments/assets/72960b84-b201-429c-9c9c-d3485c88368e" />


### New Sample Data Prediction

<img width="282" height="327" alt="image" src="https://github.com/user-attachments/assets/e89f9a1a-4403-4756-a2be-be2adf67fb4a" />

<img width="316" height="342" alt="image" src="https://github.com/user-attachments/assets/d99dc34d-4e77-4e0e-8dfc-d429c4e79183" />



## RESULT
Thus the python program to develop an image classification model using transfer learning with VGG19 architecture is executed successfully.
