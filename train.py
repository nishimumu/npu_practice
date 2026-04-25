from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN

def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0:
            avg_loss = total_loss / total
            accuracy = correct / total * 100
            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx:4d}/{len(train_loader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"Accuracy: {accuracy:.2f}%"
            )

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    return avg_loss, accuracy

def evaluate(model, device, test_loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    return avg_loss, accuracy

def main():
    torch.manual_seed(42)

    device = torch.device("cpu")
    print("Device:", device)

    data_dir = Path("data")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = SmallCNN().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 3

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
        )

        test_loss, test_acc = evaluate(
            model=model,
            device=device,
            test_loader=test_loader,
        )

        print("-" * 60)
        print(f"Epoch {epoch} finished")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Accuracy: {test_acc:.2f}%")
        print("-" * 60)

    model_path = Path("mnist_small_cnn_cpu.pth")
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
