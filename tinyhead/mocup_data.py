import torch
from torch.utils.data import Dataset
import random
import numpy as np

class MockNextTokenPredictionDataset(Dataset):
    def __init__(self, num_samples, vocab_size, sequence_length):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

    def __len__(self):
        # Returns the number of samples in the dataset
        return self.num_samples

    def __getitem__(self, idx):
        # Generates a random sequence of token IDs
        token_ids = torch.randint(low=0, high=self.vocab_size, size=(self.sequence_length,))

        return torch.from_numpy(np.array(token_ids))
