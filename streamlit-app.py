import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import cv2
from torchvision import transforms
import numpy as np
import streamlit as st
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding',False)
st.title("Pengklasifikasi Lesi Kulit")
st.text("Tolong upload gambar lesi kulit (jpg/jpeg/png)")

class ResNetModel(nn.Module):
    def __init__(self, num_classes, extractor_trainable=True):
        super(ResNetModel, self).__init__()
        resnet = models.resnet34(pretrained=True)
        
        if not extractor_trainable:
            for param in resnet.parameters():
                param.requires_grad = False
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@st.cache(allow_output_mutation=True)

def load_model():
    model = ResNetModel(num_classes=6)
    model = model.load_state_dict(torch.load('resnet_weights.pth',map_location=torch.device('cpu')))
    model = model.to('cpu')
    model.eval()  # Set model to evaluation mode
    return model

def preprocess_image(uploaded_file):
    # Read the uploaded image and process it
    image = Image.open(uploaded_file)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict_image(model, image_tensor, class_mapping):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_class = list(class_mapping.keys())[predicted_index]
        return predicted_class, probabilities

# Load the model
with st.spinner("Meload model ke memori..."):
    model = load_model()

# Class mapping for predictions
class_mapping = {'Chickenpox': 0, 'Cowpox': 1, 'HFMD': 2, 'Healthy': 3, 'Measles': 4, 'Monkeypox': 5}

# Image uploader
uploaded_file = st.file_uploader("Upload foto lesi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    image_tensor = preprocess_image(uploaded_file)

    # Predict the image class
    predicted_class, probabilities = predict_image(model, image_tensor, class_mapping)

    # Display the results
    st.image(uploaded_file, caption="Gambar lesi kulit", use_column_width=True)
    st.write(f"Predicted class: {predicted_class}")
    
    # Display probabilities for each class
    st.write("Probabilities:")
    for class_name, prob in zip(class_mapping.keys(), probabilities[0]):
        st.write(f"{class_name}: {prob.item():.4f}")

else :
    st.text("Tolong upload file foto sesuai format")

