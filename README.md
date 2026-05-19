## EXP- 4 - Developing a Neural Network Classification Model using Transfer Learning

### AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

### Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model. 


## Neural Network Model
<img width="987" height="792" alt="image" src="https://github.com/user-attachments/assets/1e412a42-ced2-4b84-87ca-f2fd46388fed" />

### DESIGN STEPS
#### STEP 1: 
Import required libraries and define image transforms.
#### STEP 2: 
Load training and testing datasets using ImageFolder.
#### STEP 3: 
Visualize sample images from the dataset.
#### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.
#### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.
#### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

### PROGRAM
#### Name: DHARSHAN R
#### Register Number: 212224230060
```python
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
from google.colab import drive
drive.mount('/content/drive')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
!unzip -qq /content/drive/MyDrive/DeepLearning/EXP-4/chip_data.zip -d data
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
print(f"Total number of test samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset
first_image1,label =test_dataset[0]
print(f"Shape of the first image: {first_image1.shape}")

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## Step 2: Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
# write your code here
from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
# Write your code here
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, input_size=(3, 224, 224))

# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
      running_loss=0.0
      for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
      train_losses.append(running_loss/len(train_loader))  

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

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_model(model, train_loader, test_loader, num_epochs=20)

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().squeeze()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=train_dataset.classes
        )
    )

# Evaluate the model
# write your code here
test_model(model,test_loader)

def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output)
        predicted = (prob > 0.5).int().squeeze().item()
    class_names = dataset.classes
    image = image.cpu().numpy().transpose((1, 2, 0))
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    image_to_display = image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(
        f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}'
    )
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)

#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)
```

### OUTPUT

#### Training Loss, Validation Loss Vs Iteration Plot
<img width="636" height="478" alt="Screenshot 2026-05-19 104010" src="https://github.com/user-attachments/assets/7f1ed769-894a-4b1f-a03f-77d890ffe2d6" />



#### Confusion Matrix
<img width="581" height="509" alt="image" src="https://github.com/user-attachments/assets/6526d4e3-88a6-47b8-8afa-71ce01db9d83" />



#### Classification Report
<img width="404" height="147" alt="image" src="https://github.com/user-attachments/assets/b4a0e994-a219-4a37-b8d2-ffbbd425371b" />


#### New Sample Data Prediction
<img width="413" height="389" alt="image" src="https://github.com/user-attachments/assets/24f90fdf-c4e5-4128-990d-3b80adb558e9" /> 
<img width="404" height="372" alt="Screenshot 2026-05-19 103854" src="https://github.com/user-attachments/assets/a857bc82-d103-4389-bfca-7ac4bfb83358" />









### RESULT
VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.
