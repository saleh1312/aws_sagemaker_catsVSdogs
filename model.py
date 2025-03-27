import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtremeTinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ExtremeTinyCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)  # 2 filters
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)  # 4 filters
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)  # 8 filters

        self.pool = nn.MaxPool2d(2, 2)  # Reduce spatial size by half
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce to (1,1) feature map
        self.fc = nn.Linear(8, num_classes)  # Fully connected layer (only 8 params per class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool
        
        x = self.global_avg_pool(x)  # Convert to (batch_size, 8, 1, 1)
        x = torch.flatten(x, 1)  # Flatten before FC
        x = self.fc(x)  # Fully Connected Layer
        return x

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExtremeTinyCNN(num_classes=2).to(device)

# Count Parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}")
