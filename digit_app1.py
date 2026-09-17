import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.transforms as transforms

import torch
from torch import nn
st.title("Digit Recognizer")

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

# Load model
model = DigitModel()

model.load_state_dict(
    torch.load(
        "digit_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, width=200)

    # invert image
    #image = ImageOps.invert(image)

    # transform
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)

    # prediction
    with torch.inference_mode():

        prediction = model(image_tensor)

        predicted_digit = prediction.argmax(dim=1).item()

    st.write(f"Predicted Digit: {predicted_digit}")
