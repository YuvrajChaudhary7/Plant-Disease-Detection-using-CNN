import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image

# Load dataset to get correct class order
dataset = datasets.ImageFolder("dataset")
classes = dataset.classes


class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(3,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3)

        self.fc1 = nn.Linear(64*30*30,128)
        self.fc2 = nn.Linear(128,len(classes))

    def forward(self,x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1,64*30*30)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Load trained model
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


def predict_image(path):

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    outputs = model(img)

    # Temperature scaling to reduce overconfidence
    temperature = 2.5
    probabilities = torch.softmax(outputs / temperature, dim=1)

    confidence, predicted = torch.max(probabilities,1)

    confidence = confidence.item()*100
    disease = classes[predicted.item()]

    # Detect random images / non leaf
    if confidence < 75:
        disease = "Unknown / Not a Tomato Leaf"

    confidence = round(confidence,2)

    # Format disease name nicely
    disease = disease.replace("_"," ")

    return disease, confidence
