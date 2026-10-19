"""Train the CNN on MNIST and save the weights to model.pth.

Usage:  python train.py [--epochs 8]
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN

MEAN, STD = 0.1307, 0.3081


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Augmentation makes the model tolerant to hand-drawn variation.
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])

    train_set = datasets.MNIST("data", train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
        acc = 100 * correct / len(test_set)
        print(f"Epoch {epoch}/{args.epochs}  loss {running / len(train_set):.4f}  test acc {acc:.2f}%")

    torch.save(model.state_dict(), "model.pth")
    print("Saved model.pth")


if __name__ == "__main__":
    main()
