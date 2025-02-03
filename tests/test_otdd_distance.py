# tests/test_otdd_distance.py
import torch
import numpy as np
from otdd.pytorch.distance import FeatureCost, DatasetDistance
from otdd.pytorch.datasets import load_vision_data

def dummy_embedding(x):
    # A dummy embedding function that simply returns flattened input.
    return x.view(x.size(0), -1)

def test_cost_matrix():
    # Create dummy source and target data.
    src_data = torch.randn(10, 3, 224, 224)
    tgt_data = torch.randn(15, 3, 224, 224)
    cost_func = FeatureCost(src_embedding=dummy_embedding, src_dim=(3,224,224),
                            tgt_embedding=dummy_embedding, tgt_dim=(3,224,224),
                            p=2, device='cpu')
    cost_matrix = cost_func.compute_cost_matrix(src_data, tgt_data)
    assert cost_matrix.shape == (10, 15)
    # Ensure that the cost matrix is nonnegative.
    assert torch.all(cost_matrix >= 0)

def test_dataset_distance():
    # Use the vision data loader to generate dummy data.
    train_loader, test_loader = load_vision_data(batch_size=32)
    cost_func = FeatureCost(src_embedding=dummy_embedding, src_dim=(3,224,224),
                            tgt_embedding=dummy_embedding, tgt_dim=(3,224,224),
                            p=2, device='cpu')
    dataset_dist = DatasetDistance(train_loader, test_loader,
                                   inner_ot_method='exact',
                                   debiased_loss=True,
                                   feature_cost=cost_func,
                                   λ_x=1.0, λ_y=1.0,
                                   sqrt_method='spectral',
                                   sqrt_niters=10,
                                   precision='single',
                                   p=2, entreg=1e-2,
                                   device='cpu')
    ot_dist, coupling = dataset_dist.distance(maxsamples=100, return_coupling=True)
    # Check that ot_dist is a float and coupling is a numpy array.
    assert isinstance(ot_dist, float)
    assert isinstance(coupling, np.ndarray)

