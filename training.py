from loaddata import *
from model import * 

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

# PLOTTING THE TRAINING
def plot_training_history(history):
    """Plots the training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss') 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss') 
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy') 
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy') 
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# EVALUATNG THE MODEL
def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluates the model on a given dataloader (e.g., test set).

    Computes confusion matrix and classification report.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate classification report
    class_report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # Calculate overall accuracy from the report
    accuracy = np.trace(cm) / np.sum(cm)  # Simple accuracy from confusion matrix

    return {
        'confusion_matrix': cm,
        'classification_report': class_report,
        'accuracy': accuracy,
        'predictions': all_preds,
        'true_labels': all_labels
    }

# VISUALIZING A COFUSION MATRIX
def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
        
    # Instantiate the model
    model = PneumoniaCNN()

    loaddata = LoadData()
    train_loader = loaddata.train_loader
    val_loader = loaddata.val_loader
    test_loader = loaddata.test_loader


    # Check if CUDA (GPU support) is available, otherwise use CPU
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MAC SPECIFIC
    # MPS is mac specific so it'll train faster on macs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Move the model to the chosen device (GPU or CPU)
    model.to(device)

    print(f"Model '{type(model).__name__}' instantiated and moved to '{device}'.")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Loss function and optimizer defined.")


    
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

    # Plot the training and validation history
    plot_training_history(history)

    # Evaluate the model
    eval_results = evaluate_model(model, test_loader, device, loaddata.class_names)

    # Print results
    print("Classification Report:")
    print(eval_results['classification_report'])
    print(f"\nOverall Accuracy: {eval_results['accuracy']:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(eval_results['confusion_matrix'], loaddata.class_names)




    # SAVING THE MODEL
    modelpath = 'model.pth'
    torch.save(model, modelpath)
    print("Model Saved to " + modelpath)
