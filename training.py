from loaddata import *
from model import * 

# Instantiate the model
model = PneumoniaCNN()

loaddata = LoadData()
train_loader = loaddata.train_loader
val_loader = loaddata.val_loader
test_loader = loaddata.test_loader


# Check if CUDA (GPU support) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MAC SPECIFIC
# MPS is mac specific so it'll train faster on macs
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move the model to the chosen device (GPU or CPU)
model.to(device)

print(f"Model '{type(model).__name__}' instantiated and moved to '{device}'.")

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Loss function and optimizer defined.")


# TRAINING FUNCTION
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    """Trains and validates the model."""
    # Initialize lists to track metrics
    train_losses = []
    val_losses = []         
    train_accuracies = []
    val_accuracies = []     

    print("Starting Training...")
    # Training loop
    for epoch in range(num_epochs):
        # Training Phase
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Iterate over training data
        for i, (images, labels) in enumerate(train_loader):
            # Move data to the specified device
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            running_loss += loss.item() * images.size(0)  # loss.item() is the avg loss per batch
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate training statistics for the epoch
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation Phase
        model.eval()  # Set model to evaluation mode (disables dropout, uses running stats for batch norm)
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Disable gradient calculations for validation
        with torch.no_grad():
            for images, labels in val_loader: 
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate validation statistics for the epoch
        epoch_val_loss = val_loss / len(val_loader.dataset)  
        epoch_val_acc = correct_val / total_val              
        val_losses.append(epoch_val_loss)                    
        val_accuracies.append(epoch_val_acc)                 

        # Print statistics for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_val_acc:.4f}") 
        print("-" * 30)

    print("Finished Training.")
    # Return performance history
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,        
        'val_accuracies': val_accuracies 
    }

# TRAINING THE MODEL
#
# Train the model
num_epochs = 20

loaddata = LoadData()

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs
)