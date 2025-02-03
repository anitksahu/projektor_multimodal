# otdd/pytorch/datasets.py

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

def load_vision_data(batch_size: int = 32):
    """
    Loads a vision dataset.
    For demonstration, generates random images with shape (3,224,224).

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    num_train = 1000
    num_test = 200
    train_data = torch.randn(num_train, 3, 224, 224)
    test_data = torch.randn(num_test, 3, 224, 224)
    train_labels = torch.randint(0, 10, (num_train,))
    test_labels = torch.randint(0, 10, (num_test,))
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class VisionLanguageDataset(Dataset):
    """
    A simple dataset class for vision-language data.
    Each sample is a tuple (data, label) where data is a dict with keys:
        'image': a tensor of shape (3,224,224)
        'text': a dict with tokenized text (e.g., 'input_ids' and 'attention_mask')
    """
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.texts = {
            "input_ids": torch.randint(0, 1000, (num_samples, 16)),
            "attention_mask": torch.ones(num_samples, 16, dtype=torch.long)
        }
        self.labels = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = {"image": self.images[idx],
                "text": {k: v[idx] for k, v in self.texts.items()}}
        label = self.labels[idx]
        return data, label

def load_vision_language_data(batch_size: int = 32):
    """
    Loads a vision-language dataset.
    For demonstration, generates random multimodal data.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    num_train = 1000
    num_test = 200
    train_dataset = VisionLanguageDataset(num_samples=num_train)
    test_dataset = VisionLanguageDataset(num_samples=num_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

