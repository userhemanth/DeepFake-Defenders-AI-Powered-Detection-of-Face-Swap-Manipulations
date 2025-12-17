import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import glob
from tqdm import tqdm

# Ensure to print the device being used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 0.001
checkpoint_dir = 'checkpoints/'

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Dataset class
class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Organize dataset into 'real' and 'fake' directories for both train and val
        self.real_images_train = glob.glob(os.path.join(data_dir, 'train', 'real', '*.jpg'))
        self.fake_images_train = glob.glob(os.path.join(data_dir, 'train', 'fake', '*.jpg'))
        self.real_images_val = glob.glob(os.path.join(data_dir, 'val', 'real', '*.jpg'))
        self.fake_images_val = glob.glob(os.path.join(data_dir, 'val', 'fake', '*.jpg'))

        # Print the number of images found in each directory
        print(f"Found {len(self.real_images_train)} real images in training.")
        print(f"Found {len(self.fake_images_train)} fake images in training.")
        print(f"Found {len(self.real_images_val)} real images in validation.")
        print(f"Found {len(self.fake_images_val)} fake images in validation.")

        # Check if both directories have images
        if len(self.real_images_train) == 0 or len(self.fake_images_train) == 0 or len(self.real_images_val) == 0 or len(self.fake_images_val) == 0:
            raise ValueError("The 'real' or 'fake' directories are empty or the paths are incorrect.")

    def __len__(self):
        # Total images from both train and val
        return len(self.real_images_train) + len(self.fake_images_train) + len(self.real_images_val) + len(self.fake_images_val)

    def __getitem__(self, idx):
        if idx < len(self.real_images_train):
            img_path = self.real_images_train[idx]
            label = 0  # Real class
        elif idx < len(self.real_images_train) + len(self.fake_images_train):
            img_path = self.fake_images_train[idx - len(self.real_images_train)]
            label = 1  # Fake class
        elif idx < len(self.real_images_train) + len(self.fake_images_train) + len(self.real_images_val):
            img_path = self.real_images_val[idx - len(self.real_images_train) - len(self.fake_images_train)]
            label = 0  # Real class
        else:
            img_path = self.fake_images_val[idx - len(self.real_images_train) - len(self.fake_images_train) - len(self.real_images_val)]
            label = 1  # Fake class

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Model definition
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        # Use pretrained ViT from torchvision
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)  # Use weights from torchvision

        # Adjust the final fully connected layer to match the number of classes (2: real or fake)
        in_features = self.vit.heads[0].in_features  # The first layer of heads gives us the in_features
        self.vit.heads = nn.Linear(in_features, 2)  # Replace the final head to have 2 classes

    def forward(self, x):
        return self.vit(x)

# Training function with checkpointing
def train_model(data_dir):
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloaders
    dataset = VideoDataset(data_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = HybridModel().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        for images, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = (correct / total) * 100

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save checkpoint for this epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        print(f"Checkpoint for epoch {epoch+1} saved.")

    print('Training completed.')

# Main execution
if __name__ == "__main__":
    data_dir = 'C:/DeepFake_Defenders/organized_frames'  # Update to your dataset location
    train_model(data_dir)
