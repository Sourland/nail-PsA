import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models

# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Imagenet stats
])

# Create datasets
train_dataset = ImageFolder('../dataset/NailCheck/train', transform=transform)
valid_dataset = ImageFolder('../dataset/NailCheck/train', transform=transform)
test_dataset = ImageFolder('../dataset/NailCheck/train', transform=transform)

# Create dataloaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_classes = len(train_dataset.classes)  # get number of classes

mobilenet = models.mobilenet_v2(pretrained=True)  # load a pretrained model
mobilenet.classifier[1] = torch.nn.Linear(mobilenet.last_channel, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mobilenet.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    mobilenet.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = mobilenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    mobilenet.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = mobilenet(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {valid_loss / len(valid_loader)}")
