import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- Config ---
DATA_DIR = "data"
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def train():
    print("📁 Loading datasets...")
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(train_dataset.classes)
    print(f"✅ Loaded {len(train_dataset)} training images, {len(val_dataset)} validation images")
    print(f"🧬 Found {num_classes} classes:")
    for idx, label in enumerate(train_dataset.classes):
        print(f"  {idx:2}: {label}")

    print("🧠 Setting up model...")
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("🚀 Starting training...\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total, correct, train_loss = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"✅ Epoch [{epoch + 1}/{NUM_EPOCHS}] - Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}\n")

    torch.save(model.state_dict(), "breed_classifier_resnet18.pt")
    print("💾 Model saved as breed_classifier_resnet18.pt")


if __name__ == '__main__':
    print("🔧 Starting training script...\n")
    train()
