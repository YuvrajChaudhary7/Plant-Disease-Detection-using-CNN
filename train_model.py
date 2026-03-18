import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN Model
class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64*30*30,128)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self,x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1,64*30*30)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = CNN(len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):

    running_loss = 0

    for images, labels in loader:

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")

# Save model
torch.save(model.state_dict(), "model.pth")

print("Training complete")