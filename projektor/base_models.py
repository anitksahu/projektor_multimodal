# projektor/base_models.py
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

class BaseClassifier(ABC, nn.Module):
    """
    Base classifier interface.
    Any classifier you wish to use should inherit from this class and implement forward().
    """
    def __init__(self, num_classes):
        super(BaseClassifier, self).__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """Compute the forward pass."""
        pass

    def train_model(self, train_loader, epochs, device, optimizer=None, criterion=None, verbose=True):
        """
        Generic training loop.
        Returns the best (lowest) training loss over epochs.
        """
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=0.001)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        best_train_loss = float('inf')
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            total_samples = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                batch_size = inputs.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size
            avg_loss = epoch_loss / total_samples
            best_train_loss = min(best_train_loss, avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"[Epoch {epoch:03d}] Train Loss: {avg_loss:.3f}")
        return best_train_loss

    def evaluate(self, loader, device, criterion=None):
        """
        Generic evaluation loop.
        Returns average loss and accuracy (in percentage).
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total += batch_size
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy

class BaseEmbedder(ABC, nn.Module):
    """
    Base embedder interface.
    Any model used for computing embeddings (for OT distance, etc.)
    should inherit from this class and implement forward().
    """
    @abstractmethod
    def forward(self, x):
        """Return the embedding of x."""
        pass

