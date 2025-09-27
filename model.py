import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduce spatial dimensions by half
        )

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Further reduce spatial dimensions
        )

        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Further reduce spatial dimensions
        )

        # Flatten layer to convert 3D feature maps to 1D vector
        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=128 * 32 * 32, out_features=512)  # Adjust size based on input dimensions
        self.dropout1 = nn.Dropout(0.5)  # Add 50% dropout for regularization
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=128, out_features=2)  # 2 output classes: Normal and Pneumonia

    def forward(self, x):
        # Pass input through convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Flatten the features
        x = self.flatten(x)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)  # Raw, unnormalized predictions

        return logits
