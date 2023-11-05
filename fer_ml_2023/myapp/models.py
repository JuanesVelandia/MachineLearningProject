from django.db import models
from django.conf import settings
import torch, os, io
from torch import nn
from torchvision import transforms as T
from PIL import Image


class FacialExpressionCNN(nn.Module):
    def __init__(self, num_classes):
        super(FacialExpressionCNN, self).__init__()

        # Capas convolucionales
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2, stride=2))

        # Capas fully connected
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3))

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten the fully-connected layer
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Clase para manejar el modelo
class FacialExpressionModel:
    def __init__(self, model_path):
        self.classes = ['angry','disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = FacialExpressionCNN(len(self.classes))

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, image_file):
        transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((48, 48)),
            T.ToTensor(),
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = Image.open(image_file)

        image = transform(image)

        to_pil = T.ToPILImage()
        transformed_image = to_pil(image)

        img_byte_array = io.BytesIO()
        transformed_image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
 
        file_name = "img_tra.png" 
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        with open(file_path, "wb") as f:
            f.write(img_byte_array)


        image = image.unsqueeze(0)

        output = self.model(image)
        _, predicted = torch.max(output, 1)
        return self.classes[predicted.item()]

