# otdd/pytorch/distance.py

import torch
import torch.nn.functional as F
import numpy as np
import ot  # POT: Python Optimal Transport

class FeatureCost:
    def __init__(self, src_embedding, src_dim, tgt_embedding, tgt_dim, p=2, device='cuda'):
        """
        Initializes a cost function to compute feature distances between two datasets.

        Parameters:
            src_embedding: Callable that extracts features from the source data.
            src_dim: Expected dimension (or description) of the source features.
            tgt_embedding: Callable that extracts features from the target data.
            tgt_dim: Expected dimension of the target features.
            p (int): The norm to use (e.g., 2 for Euclidean).
            device (str): The device for computation.
        """
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.p = p
        self.device = device

    def compute_cost_matrix(self, src_data, tgt_data):
        """
        Computes a pairwise cost matrix between source and target samples.

        Parameters:
            src_data: A batch of source samples.
            tgt_data: A batch of target samples.

        Returns:
            torch.Tensor: A cost matrix of shape (n_src, n_tgt) where each entry is the p-norm distance.
        """
        # Obtain features using the provided embedding functions.
        src_features = self.src_embedding(src_data)  # shape: (n_src, d)
        tgt_features = self.tgt_embedding(tgt_data)  # shape: (n_tgt, d)

        # Flatten features if necessary.
        src_features = src_features.view(src_features.size(0), -1)
        tgt_features = tgt_features.view(tgt_features.size(0), -1)

        # Compute pairwise distances.
        n_src = src_features.size(0)
        n_tgt = tgt_features.size(0)
        # Use broadcasting to compute pairwise differences
        src_exp = src_features.unsqueeze(1).expand(n_src, n_tgt, -1)
        tgt_exp = tgt_features.unsqueeze(0).expand(n_src, n_tgt, -1)
        cost = torch.norm(src_exp - tgt_exp, p=self.p, dim=2)
        return cost

class DatasetDistance:
    def __init__(self, src_loader, tgt_loader, inner_ot_method='exact', debiased_loss=True,
                 feature_cost=None, λ_x=1.0, λ_y=1.0, sqrt_method='spectral', sqrt_niters=10,
                 precision='single', p=2, entreg=1e-2, device='cuda'):
        """
        Computes the OT distance between two datasets.

        Parameters:
            src_loader: DataLoader for the source dataset.
            tgt_loader: DataLoader for the target dataset.
            inner_ot_method (str): The method to compute the inner OT (e.g., 'exact').
            debiased_loss (bool): Whether to use debiased loss.
            feature_cost (FeatureCost): An instance of the FeatureCost class.
            λ_x, λ_y: Regularization parameters.
            sqrt_method (str): Method for computing matrix square roots.
            sqrt_niters (int): Number of iterations for square root computation.
            precision (str): 'single' or 'double' precision.
            p (int): p-norm for the cost.
            entreg (float): Tolerance for convergence.
            device (str): Device identifier.
        """
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.inner_ot_method = inner_ot_method
        self.debiased_loss = debiased_loss
        self.feature_cost = feature_cost
        self.λ_x = λ_x
        self.λ_y = λ_y
        self.sqrt_method = sqrt_method
        self.sqrt_niters = sqrt_niters
        self.precision = precision
        self.p = p
        self.entreg = entreg
        self.device = device

    def distance(self, maxsamples=5000, return_coupling=False):
        """
        Computes the OT distance between the source and target datasets.

        This method samples up to maxsamples from each dataset, computes the cost matrix,
        and then uses an optimal transport solver (from the POT library) to compute the
        optimal coupling. The OT distance is taken as the sum of elementwise products between
        the cost matrix and the coupling matrix.

        Parameters:
            maxsamples (int): Maximum number of samples to use from each dataset.
            return_coupling (bool): If True, returns the coupling matrix as well.

        Returns:
            ot_distance (float) or tuple: (ot_distance, coupling) if return_coupling is True.
        """
        # Sample a subset from the source loader.
        src_samples = []
        for batch in self.src_loader:
            data = batch[0]
            if isinstance(data, dict):  # if multimodal, assume 'image' key is used for embedding.
                data = data["image"]
            src_samples.append(data)
            if torch.cat(src_samples).size(0) >= maxsamples:
                break
        # Similarly for the target loader.
        tgt_samples = []
        for batch in self.tgt_loader:
            data = batch[0]
            if isinstance(data, dict):
                data = data["image"]
            tgt_samples.append(data)
            if torch.cat(tgt_samples).size(0) >= maxsamples:
                break

        src_samples = torch.cat(src_samples)[:maxsamples].to(self.device)
        tgt_samples = torch.cat(tgt_samples)[:maxsamples].to(self.device)

        cost_matrix = self.feature_cost.compute_cost_matrix(src_samples, tgt_samples)
        # Convert cost matrix to numpy for POT.
        cost_np = cost_matrix.cpu().detach().numpy()

        # Create uniform distributions over source and target samples.
        n_src = cost_np.shape[0]
        n_tgt = cost_np.shape[1]
        a = np.ones(n_src) / n_src
        b = np.ones(n_tgt) / n_tgt

        # Compute the optimal coupling using the Earth Mover's Distance solver.
        coupling = ot.emd(a, b, cost_np)
        ot_distance = np.sum(coupling * cost_np)

        if return_coupling:
            return ot_distance, coupling
        return ot_distance

