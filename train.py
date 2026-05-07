import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
# Dataset
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=32
)

# Model
class DigitModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.classifier(x)

        return x

# Create model
model = DigitModel()

# Loss and optimizer
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 5

for epoch in range(epochs):

    model.train()

    for X, y in train_dataloader:

        y_pred = model(X)

        loss = loss_function(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1} completed")

# Save model
torch.save(model.state_dict(), "digit_model.pth")

print("Model Saved Successfully")