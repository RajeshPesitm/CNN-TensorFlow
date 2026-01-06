import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys

# -----------------------------
# 1️⃣ Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2️⃣ Transforms
# -----------------------------
# MNIST and EMNIST are grayscale 28x28
transform = transforms.Compose([
    transforms.ToTensor()
])

# -----------------------------
# 3️⃣ Load MNIST for training
# -----------------------------
# Rotate EMNIST digits to match MNIST orientation
emnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.transpose(1, 2))   # comment / Uncomment
])
train_dataset = datasets.EMNIST(root="data", split="digits", train=True, download=True, transform=emnist_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# -----------------------------
# 4️⃣ Load MNIST digits for testing
# -----------------------------
test_dataset = datasets.EMNIST(root="data", split="digits", train=False, download=True, transform=emnist_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 5️⃣ Define the model (MLP)
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP().to(device)

# -----------------------------
# 6️⃣ Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 7️⃣ Train on MNIST
# -----------------------------
epochs = 8
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# -----------------------------
# 8️⃣ Evaluate on EMNIST
# -----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Inter-dataset Test Accuracy (MNIST→EMNIST): {100*correct/total:.2f}%")

# -----------------------------
# 9️⃣ Interactive loop for predictions
# -----------------------------
def on_key(event):
    if event.key == 'c':
        print("Closing loop...")
        plt.close('all')
        sys.exit()

# Convert test dataset to numpy arrays for interactive predictions
x_test = np.array([img.numpy().squeeze() for img, _ in test_dataset])
y_test = np.array([label for _, label in test_dataset])

index = 0
num_samples = len(x_test)

while True:
    img_tensor = torch.tensor(x_test[index]).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    ax.imshow(x_test[index], cmap='gray')
    ax.set_title(f"Predicted: {predicted_label}, Actual: {y_test[index]}")
    ax.axis('off')

    print(f"Image {index}: Predicted: {predicted_label}, Actual: {y_test[index]}")
    plt.show()

    index = (index + 1) % num_samples
