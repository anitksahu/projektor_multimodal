# projektor/experiment.py
"""
Experiment Runner

This script runs an end-to-end experiment, including:
  - Loading data (vision-only or multimodal).
  - Instantiating a classifier and an embedder.
  - Training and evaluating the classifier.
  - Computing the optimal transport (OT) distance between training and test sets.
  - Optionally, performing performance prediction using the Projector.

Usage examples:
--------------
For vision–only experiments:
    python experiment.py --cnum 0 --dataset_type vision --num_classes 10 --input_shape 3,224,224 --epochs 50 --batch_size 32 --embed_dim 512 --train_classifier True --outfile results.res

For vision–language experiments:
    python experiment.py --cnum 0 --dataset_type multimodal --num_classes 10 --input_shape 3,224,224 --epochs 50 --batch_size 32 --embed_dim 768 --train_classifier True --outfile results_multimodal.res
"""

import os
import argparse
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Import OTDD dataset loaders and distance functions.
from otdd.pytorch.datasets import load_vision_data, load_vision_language_data
from otdd.pytorch.distance import DatasetDistance, FeatureCost

# Import default models and multimodal embedder.
from projektor.default_models import SimpleClassifier, SimpleEmbedder, CLIPMultiModalEmbedder
# Import the Projector class (for performance prediction) if needed.
from projektor.projector import Projector

# -------------------------------------
# Training and Evaluation
# -------------------------------------
def train_and_evaluate(classifier, train_loader, test_loader, device, epochs, train_flag=True):
    """
    Trains (if requested) and evaluates the classifier.
    
    Returns:
      - loss_diff: Difference between test loss and best training loss (if training was done)
      - test_accuracy: Test accuracy as a percentage.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classifier.to(device)
    best_train_loss = float('inf')
    
    if train_flag:
        classifier.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            total = 0
            for inputs, targets in train_loader:
                # For multimodal data, use the 'image' field.
                if isinstance(inputs, dict):
                    inputs = inputs["image"]
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = classifier(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
            avg_loss = epoch_loss / total
            best_train_loss = min(best_train_loss, avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch+1:03d}] Train Loss: {avg_loss:.3f}")
    else:
        print("Skipping training; using provided classifier weights.")
    
    # Evaluation.
    classifier.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            if isinstance(inputs, dict):
                inputs = inputs["image"]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    avg_test_loss = test_loss / total
    accuracy = 100. * correct / total
    loss_diff = avg_test_loss - best_train_loss if train_flag else None
    return loss_diff, accuracy

def compute_ot_distance(embedder, train_loader, test_loader, device, embed_dim, max_samples=5000):
    """
    Computes the optimal transport (OT) distance between the training and test sets.

    Parameters:
       embedder: The embedding model used to extract features.
       embed_dim: A tuple specifying the expected dimension of the features.
       max_samples: Maximum number of samples to use from each dataset.

    Returns:
       ot_distance: The computed OT distance (a scalar).
    """
    feature_cost = FeatureCost(src_embedding=embedder, src_dim=embed_dim,
                               tgt_embedding=embedder, tgt_dim=embed_dim,
                               p=2, device=device)
    ot_distance_computer = DatasetDistance(train_loader, test_loader,
                                           inner_ot_method='exact',
                                           debiased_loss=True,
                                           feature_cost=feature_cost,
                                           λ_x=1.0, λ_y=1.0,
                                           sqrt_method='spectral',
                                           sqrt_niters=10,
                                           precision='single',
                                           p=2, entreg=1e-2,
                                           device=device)
    ot_distance, _ = ot_distance_computer.distance(maxsamples=max_samples, return_coupling=True)
    return ot_distance

# -------------------------------------
# Main Experiment Function
# -------------------------------------
def run_experiment(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cnum)
    device = torch.device(f"cuda:{args.cnum}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    # Choose data loader and models based on dataset type.
    if args.dataset_type == "vision":
        train_loader, test_loader, num_classes = load_vision_data(args.batch_size, args.num_classes, input_shape)
        classifier = SimpleClassifier(num_classes, input_shape).to(device)
        embedder = SimpleEmbedder(num_classes, input_shape).to(device)
    elif args.dataset_type == "multimodal":
        train_loader, test_loader, num_classes = load_vision_language_data(args.batch_size, args.num_classes, input_shape)
        classifier = SimpleClassifier(num_classes, input_shape).to(device)  # Replace with a vision-language classifier if available.
        embedder = CLIPMultiModalEmbedder(device)
    else:
        raise ValueError("Unsupported dataset_type. Choose 'vision' or 'multimodal'.")

    print("Running a single experiment:")
    loss_diff, test_acc = train_and_evaluate(classifier, train_loader, test_loader, device,
                                             epochs=args.epochs, train_flag=args.train_classifier)
    print(f"Classifier: Loss diff: {loss_diff:.3f}, Test Accuracy: {test_acc:.2f}%")

    ot_distance = compute_ot_distance(embedder, train_loader, test_loader, device,
                                      embed_dim=args.embed_dim, max_samples=args.max_samples)
    print(f"OT distance: {ot_distance:.3f}")

    # Optionally, run the Projector for performance prediction if projector data is provided.
    if args.projector_data:
        try:
            with open(args.projector_data, 'rb') as f:
                reserrlog, otlog, accs = pickle.load(f)
            proj = Projector(pad_length=11, threshold=5)
            aligned = proj.align_data(reserrlog)
            _, mask_high = proj.compute_masks(aligned)
            X, y = proj.prepare_features(reserrlog, otlog, accs, mask_high)
            proj.fit(X, y, verbose=True, plot=args.plot_projector)
            predicted_perf = proj.predict(X[0])
            print("Projector predicted performance for first feature vector:", predicted_perf)
        except Exception as e:
            print("Error loading projector data:", e)
    else:
        print("No projector data provided; skipping performance projection.")

    results = {"loss_diff": loss_diff, "test_accuracy": test_acc, "ot_distance": ot_distance}
    with open(args.outfile, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {args.outfile}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generalized Experiment Runner")
    parser.add_argument('--cnum', type=int, required=True, help="CUDA device number to use")
    parser.add_argument('--dataset_type', type=str, default="vision", choices=["vision", "multimodal"],
                        help="Dataset type: 'vision' or 'multimodal'")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of classes in the dataset")
    parser.add_argument('--input_shape', type=str, required=True,
                        help="Input shape as comma-separated integers (e.g., '3,224,224')")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for data loading")
    parser.add_argument('--max_samples', type=int, default=5000, help="Max samples for OT distance computation")
    parser.add_argument('--outfile', type=str, default="experiment_results.res", help="File to save results")
    parser.add_argument('--train_classifier', type=lambda s: s.lower() in ['true', '1'], default=True,
                        help="Whether to train the classifier")
    parser.add_argument('--embed_dim', type=str, default="512",
                        help="Embedding dimension as comma-separated integers (e.g., '512')")
    parser.add_argument('--projector_data', type=str, default="",
                        help="File containing pre-computed projector data (optional)")
    parser.add_argument('--plot_projector', type=lambda s: s.lower() in ['true', '1'], default=False,
                        help="Whether to plot projector fitting results")
    return parser.parse_args()

def parse_embed_dim(s):
    return tuple(int(x) for x in s.split(","))

if __name__ == '__main__':
    args = parse_args()
    args.embed_dim = parse_embed_dim(args.embed_dim)
    run_experiment(args)

