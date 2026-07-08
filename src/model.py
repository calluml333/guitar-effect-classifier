"""Model definitions for classifier head.

Provides a simple MLP classifier that accepts embeddings or flattened
spectrogram inputs and outputs logits for the effect classes.
"""
from typing import Optional

import torch
import torch.nn as nn


class EffectClassifier(nn.Module):
	"""Simple classifier head.

	Args:
		input_dim: dimension of input features (embedding dim or flattened mel)
		n_classes: number of effect classes
		hidden_dim: size of hidden layer
		dropout: dropout probability
	"""

	def __init__(self, input_dim: int, n_classes: int = 7, hidden_dim: int = 512, dropout: float = 0.3):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim // 2, n_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def build_classifier_from_dataset_sample(sample_tensor: torch.Tensor, n_classes: int = 7) -> nn.Module:
	"""Helper to instantiate classifier with input dim inferred from a sample.

	If sample is a spectrogram (2D), flatten it.
	"""
	if sample_tensor.dim() > 1:
		input_dim = sample_tensor.numel()
	else:
		input_dim = sample_tensor.shape[0]
	return EffectClassifier(input_dim=input_dim, n_classes=n_classes)

