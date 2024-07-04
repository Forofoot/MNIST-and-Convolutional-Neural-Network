import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super(ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(n_kernels, n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(n_kernels * 4 * 4, 50),
            nn.ReLU(),
            nn.Linear(50, output_size)
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_size, n_hidden, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.net(x)

def train(model, train_loader, perm, n_epoch=1):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())

    for epoch in range(n_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Step: {batch_idx}, Loss: {loss.item()}')

def test(model, test_loader, perm):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)
            
            logits = model(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    print(f'Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

def main():
    perm = torch.arange(0, 784).long()

    tf = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(datasets.MNIST("../data/raw", download=True, train=True, transform=tf),
    batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST("../data/raw", download=True, train=False, transform=tf),
    batch_size=64, shuffle=True)

    input_size = 28 * 28
    output_size = 10
    n_hidden = 8

    mlp = MLP(input_size, n_hidden, output_size)
    mlp.to(device)

    print(f"Parameters (MLP) = {sum(p.numel() for p in mlp.parameters()) / 1e3}K")
    train(mlp, train_loader, perm, n_epoch=5)
    test(mlp, test_loader, perm)

    n_kernels = 6
    convnet = ConvNet(input_size=(28, 28), n_kernels=n_kernels, output_size=output_size)
    convnet.to(device)

    print(f"Parameters (CNN) = {sum(p.numel() for p in convnet.parameters()) / 1e3}K")
    train(convnet, train_loader, perm, n_epoch=5)
    test(convnet, test_loader, perm)

    torch.save(convnet.state_dict(), "model/mnist-0.0.1.pt")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()