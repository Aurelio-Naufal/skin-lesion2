import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import cv2
from torchvision import transforms
import numpy as np

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
    
# model = ResNetModel(num_classes=6)

# model.load_state_dict(torch.load('model_weights.pth'))

# model = model.to('cpu')

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image at path: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def predict_image(model, image_tensor, class_mapping):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_class = [class_name for class_name, class_index in class_mapping.items() if class_index == predicted_index][0]
        return predicted_class, probabilities

if __name__ == "__main__":
    class_mapping = {'Chickenpox': 0, 'Cowpox': 1, 'HFMD': 2, 'Healthy': 3, 'Measles': 4, 'Monkeypox': 5}

    model = ResNetModel(num_classes=len(class_mapping))
    model.load_state_dict(torch.load('resnet_weights.pth'))
    model = model.to('cpu')

    image_path = './dataset/Valid/Chickenpox/CHP_06_01.jpg' # ganti jadi path ke gambar

    image_tensor = preprocess_image(image_path)

    predicted_class, probabilities = predict_image(model, image_tensor, class_mapping)

    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities}")