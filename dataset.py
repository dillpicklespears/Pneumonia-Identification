import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import numpy as np
import tarfile
import os
import collections
import random
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from PIL import Image
import seaborn as sns

class XRayDataset(Dataset):
    """Custom Dataset for loading X-ray images from file paths."""
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of paths to images.
            labels (list): List of corresponding labels (0 or 1).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        #Fetches the sample at the given index, loads the image,
        #applies transformations, and handles potential errors.

        #Args:
        #    idx (int): The index of the sample to fetch.

        #Returns:
        #    tuple: (image_tensor, label) if successful.
        #    None: If an error occurs (e.g., file not found, processing error),
        #          signalling to skip this sample.
        
        # Get the path and label for the requested index
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Load the image using PIL within a context manager
            with Image.open(img_path) as img:
                # Apply transforms ONLY if they exist
                if self.transform:
                    # Apply the entire transform pipeline
                    image_tensor = self.transform(img)
                    # Return the processed tensor and label
                    return image_tensor, label
                else:
                    # This branch indicates a setup error, as the transform
                    # pipeline should at least contain ToTensor().
                    raise ValueError(f"Dataset initialized without transforms for {img_path}. "
                                     "Transforms (including ToTensor) are required.")

        except FileNotFoundError:
            # Handle cases where the image file doesn't exist
            print(f"Warning: Image file not found at {img_path}. Skipping sample {idx}.")
            return None  # Returning None signals to skip
        except ValueError as e:
            # Catch the specific error we raised for missing transforms
             print(f"Error for sample {idx} at {img_path}: {e}")
             raise e  # Re-raise critical setup errors
        except Exception as e:
            # Catch any other PIL loading or transform errors
            print(f"Warning: Error processing image {img_path} (sample {idx}): {e}. Skipping sample.")
            return None  # Returning None signals to skip