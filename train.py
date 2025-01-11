import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm


from data import create_dataset, create_dataloaders





class ArtworkClassifier(nn.Module):
    def __init__(self, num_artists):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, num_artists),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

class ArtistPerformanceTracker:
    def __init__(self, num_artists):
        self.num_artists = num_artists
        self.reset()

    def reset(self):
        self.correct = np.zeros(self.num_artists)
        self.total = np.zeros(self.num_artists)

    def update(self, predictions, labels):
        for pred, label in zip(predictions, labels):
            self.total[label] += 1
            if pred == label:
                self.correct[label] += 1

    def get_accuracies(self):
        return {i: (self.correct[i] / self.total[i] if self.total[i] > 0 else 0)
                for i in range(self.num_artists)}


def train_epoch(model, train_loader, criterion, optimizer, device, performance_tracker):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update per-artist performance
        performance_tracker.update(predicted.cpu().numpy(), labels.cpu().numpy())

        if i % 10 == 0:
            print(f'Batch {i}: Loss = {loss.item():.4f}, Accuracy = {100 * correct / total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, performance_tracker):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update per-artist performance
            performance_tracker.update(predicted.cpu().numpy(), labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset, artist_names = create_dataset(args)
    train_loader, val_loader = create_dataloaders(args, train_dataset, val_dataset)

    # Create model and training components
    model = ArtworkClassifier(num_artists=len(artist_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create performance trackers
    train_tracker = ArtistPerformanceTracker(len(artist_names))
    val_tracker = ArtistPerformanceTracker(len(artist_names))

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Reset performance trackers
        train_tracker.reset()
        val_tracker.reset()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, train_tracker
        )
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

        # Log per-artist training performance
        train_accuracies = train_tracker.get_accuracies()
        print("\nPer-artist training accuracies:")
        for idx, acc in train_accuracies.items():
            artist = train_dataset.get_artist_name(idx)
            print(f"{artist}: {acc*100:.1f}%")

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, val_tracker
        )
        print(f"\nValidation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Log per-artist validation performance
        val_accuracies = val_tracker.get_accuracies()
        print("\nPer-artist validation accuracies:")
        for idx, acc in val_accuracies.items():
            artist = val_dataset.get_artist_name(idx)
            print(f"{artist}: {acc*100:.1f}%")

        # Save checkpoint if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'artist_to_idx': train_dataset.artist_to_idx,
                'per_artist_accuracy': val_accuracies
            }
            torch.save(checkpoint, f'checkpoints/best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")



if __name__ == "__main__":
    import argparse
    test_args = [
        '--artist_file', '/content/artists.txt',
        '--batch_size', '32',
        '--epochs', '50',
        '--learning_rate', '1e-3',
        '--num_workers', '4',
        '--seed', '42',
        '--max_samples_per_artist', 'None'
    ]

    # if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train artwork classifier")
    parser.add_argument('--artist_file', type=str, default='artists.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples_per_artist', type=lambda x: None if x.lower() == 'none' else int(x), default=None)

    args = parser.parse_args(test_args)
    os.makedirs("checkpoints", exist_ok=True)
    main(args)

