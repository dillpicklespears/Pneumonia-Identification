# following along: 
# https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

# All TorchVision datasets have two parameters 
# -transform to modify the features 
# -target_transform to modify the labels 
# they accept callables containing the transformation logic

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# The FashionMNIST features are in PIL Image format, 
# and the labels are integers. 
# For training, we need the features as normalized tensors, 
# and the labels as one-hot encoded tensors. 
# To make these transformations, we use ToTensor and Lambda.

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# TOTENSOR 
# ---------------------

# ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. 
# and scales the image’s pixel intensity values in the range [0., 1.]


# LAMBDA TRANSFORMS 
# ---------------------

# Lambda transforms apply any user-defined lambda function. 
# Here, we define a function to turn the integer into a one-hot encoded tensor

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))