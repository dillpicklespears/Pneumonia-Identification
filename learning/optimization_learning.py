# following along: 
# https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

# Training a model is an iterative process: 
# in each iteration the model makes a guess about the output, 
# calculates the error in its guess (loss), 
# collects the derivatives of the error with respect to its parameters (as we saw in the previous section), 
# and optimizes these parameters using gradient descent

# the article recommends this video and i also highly recommend it if 
# backproagation / gradient descent is still kind of lost on you: 
# https://www.youtube.com/watch?v=tIeHLnjs5U8


# PREREQUISTE CODE
# --------------------- 

# code from the previous sections on Datasets & DataLoaders and Build Model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


# HYPERPARAMETERS 
# --------------------- 

# Hyperparameters are adjustable parameters that let you control the model optimization process. 
# Different hyperparameter values can impact model training and convergence rates

# We define the following hyperparameters for training:
# * Number of Epochs - the number of times to iterate over the dataset
# * Batch Size - the number of data samples propagated through the network before the parameters are updated
# * Learning Rate - how much to update models parameters at each batch/epoch. 
#   Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

# in simple terms: 

learning_rate = 1e-3 # the effect the dataset has on the model each epoch / batch
# higher: might jump around too much as each set could affect the model heavily, but trains faster
# lower: train slower as each set only affects the model slightly, but could be more accurate
batch_size = 64 # the number of samples before we update the model 
# larger: risk overfitting, slow down training but more stable updates
# smaller: less memory usage, better generalization, speed up traiing but requires more iterations
epochs = 5 # the number of times to iterate over the dataset (this is already pretty simple)
# more: overfitting 
# less: underfitting

# ^^ these are numbers we might end up changing frequently when training a model


# OPTIMIZATION LOOP
# --------------------- 

# Each iteration of the optimization loop is called an epoch.

# Each epoch consists of two main parts:
# * The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
# * The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.


# LOSS FUNCTION 
# --------------------- 

# When presented with some training data, 
# our untrained network is likely not to give the correct answer. 
# Loss function measures the degree of dissimilarity of obtained result to the target value, 
# and it is the loss function that we want to minimize during training. 
# To calculate the loss we make a prediction using the inputs of our given data sample 
# and compare it against the true data label value.

# Common loss functions include 
# nn.MSELoss (Mean Square Error) for regression tasks, and 
# nn.NLLLoss (Negative Log Likelihood) for classification. 
# nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()


# OPTIMIZER 
# --------------------- 

# Optimization is the process of adjusting model parameters to 
# reduce model error in each training step. 
# Optimization algorithms define how this process is performed 
# (in this example we use Stochastic Gradient Descent). 
# All optimization logic is encapsulated in the optimizer object. 
# Here, we use the SGD optimizer; 
# additionally, there are many different optimizers available 
# that work better for different kinds of models and data.

# We initialize the optimizer by registering the model’s parameters that need to be trained, 
# and passing in the learning rate hyperparameter.

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Inside the training loop, optimization happens in three steps:
# * Call optimizer.zero_grad() to reset the gradients of model parameters. 
#   Gradients by default add up; to prevent double-counting, 
#   we explicitly zero them at each iteration.
# * Backpropagate the prediction loss with a call to loss.backward(). 
#   PyTorch deposits the gradients of the loss w.r.t. each parameter.
# * Once we have our gradients, we call optimizer.step() to adjust 
#   the parameters by the gradients collected in the backward pass.


# FULL IMPLEMENTATION
# --------------------- 

# We define train_loop that loops over our optimization code, 
# and test_loop that evaluates the model’s performance against our test data.

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# We initialize the loss function and optimizer, and pass it to train_loop and test_loop. 
# Feel free to increase the number of epochs to track the model’s improving performance.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# mine ended wtih 71.4% accuracy
# high accuracy could be acheived with more epochs
# or more tweaking of the hyperparamters 

# for saving the model reference: 
# https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

# i didn't include a file for this tutorial because its less easy to understand 
# without having a pretrained model first 

# to save this model we can do the following: 

import torchvision.models as models

torch.save(model, 'model.pth')

# models are loaded with torch.load and the path to the model