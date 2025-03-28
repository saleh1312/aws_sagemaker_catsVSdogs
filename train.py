import os
import json
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from model import model
from load_hyper_param import hyperparams

# Load environment variables
load_dotenv()

# Read directories and run ID from environment variables
data_dir = os.getenv("INPUT_DATA_PATH")
outs_dir = os.getenv("OUTPUT_DATA_PATH")
run_id = os.getenv("RUN_ID")  # Default if RUN_ID is not set

# Ensure data directory exists
if not os.path.exists(data_dir):
    raise Exception(f"Data directory not found: {data_dir}")

# Ensure data directory exists
if not run_id:
    raise Exception(f"run id not found")

# Create output directory for this run
run_outs_dir = os.path.join(outs_dir, f"{run_id}_outs")
os.makedirs(run_outs_dir, exist_ok=True)

# Save hyperparameters
hyperparams_file = os.path.join(run_outs_dir, "hyperparams.json")
with open(hyperparams_file, "w") as f:
    json.dump(hyperparams, f, indent=4)

# Hyperparameters
learning_rate = float(hyperparams["learning_rate"])
batch_size = int(hyperparams["batch_size"])
num_epochs = int(hyperparams["num_epochs"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load data
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Logging
log_file = os.path.join(run_outs_dir, "training_log.txt")

# Training Loop
train_losses, test_losses, train_accs, test_accs = [], [], [], []

with open(log_file, "w") as log:
    log.write(f"Training Run ID: {run_id}\n")
    log.write(f"Hyperparameters: {json.dumps(hyperparams, indent=4)}\n\n")

    for epoch in range(num_epochs):
        model.train()
        train_correct, train_total, train_loss = 0, 0, 0

        for images, labels in tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        # Evaluation
        model.eval()
        test_correct, test_total, test_loss = 0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, total=len(test_loader), desc=f"Evaluating epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        test_acc = test_correct / test_total
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc)

        log.write(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.4f}\n")

# Save Model
torch.save(model.state_dict(), os.path.join(run_outs_dir, "cat_vs_dog_model.pth"))

# Save Training Metrics Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Test Loss")
plt.savefig(os.path.join(run_outs_dir, "loss_plot.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accs, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), test_accs, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Test Accuracy")
plt.savefig(os.path.join(run_outs_dir, "accuracy_plot.png"))
plt.close()

# Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(os.path.join(run_outs_dir, filename))
    plt.close()

# Save Confusion Matrix
plot_confusion_matrix(y_true, y_pred, "Test Set Confusion Matrix", "test_confusion_matrix.png")

print(f"Training complete! Model, logs, and plots saved in {run_outs_dir}")
