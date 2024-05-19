import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# Function to initialize the weights of the network
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizers to compare with adjusted learning rates
optimizers = {
    'AdaGrad': lambda params: optim.Adagrad(params, lr=0.01),
    'RMSProp': lambda params: optim.RMSprop(params, lr=0.001),
    'SGDNesterov': lambda params: optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True),
    'AdaDelta': lambda params: optim.Adadelta(params, lr=1.0),
    'Adam': lambda params: optim.Adam(params, lr=0.0001)  # Adjusted learning rate for Adam
}


# Train the network and compare optimizers
def train_model(optimizer_func, epochs=20):  # Increased number of epochs
    net = Net()
    initialize_weights(net)
    optimizer = optimizer_func(net.parameters())

    net.train()
    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                losses.append(running_loss / 100)
                running_loss = 0.0

    return losses


# Plot the results
plt.figure(figsize=(10, 7))
for name, optimizer_func in optimizers.items():
    loss_values = train_model(optimizer_func)
    plt.plot(loss_values, label=name)

plt.xlabel('Iterations (in hundreds)')
plt.ylabel('Training Loss')
plt.yscale('log')
plt.legend()
plt.title('Training Loss vs. Iterations for Different Optimizers')
plt.show()
