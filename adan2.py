import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Training function
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Validation function
def validate(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(test_loader)

# Main function to train and compare optimizers
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizers = {
        'RMSprop': optim.RMSprop,
        'SGD': optim.SGD,
        'Adam': optim.Adam
    }
    num_epochs = 200
    results = {opt: {'train_loss': [], 'val_loss': []} for opt in optimizers}

    for opt_name, opt_class in optimizers.items():
        print(f"Training with {opt_name}")
        model = SimpleNN().to(device)
        optimizer = opt_class(model.parameters(), lr=0.001)  # Ensure the learning rate is set
        for epoch in range(num_epochs):
            train_loss = train(model, optimizer, criterion, train_loader, device)
            val_loss = validate(model, criterion, test_loader, device)
            results[opt_name]['train_loss'].append(train_loss)
            results[opt_name]['val_loss'].append(val_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plotting the results
    plt.figure(figsize=(12, 6))
    for opt_name in optimizers:
        plt.plot(results[opt_name]['train_loss'], label=f'{opt_name} Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison on MNIST')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

