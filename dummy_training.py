from model import *

# Create model instance
model = PneumoniaCNN()

# Create a random dummy grayscale image (batch_size, channels, height, width)
dummy_input = torch.randn(1, 1, 256, 256)

# Forward pass function with shape printing
def forward_with_shape_printing(model, x):
    print(f"Input shape: \t\t{x.shape}") # Using tabs for alignment

    # Pass through convolutional blocks
    x = model.conv_block1(x)
    print(f"After conv_block1: \t{x.shape}")
    x = model.conv_block2(x)
    print(f"After conv_block2: \t{x.shape}")
    x = model.conv_block3(x)
    print(f"After conv_block3: \t{x.shape}")

    # Flatten the features
    x = model.flatten(x)
    print(f"After flatten: \t\t{x.shape}")

    # Pass through fully connected layers (only showing final output shape)
    x = F.relu(model.fc1(x))
    x = model.dropout1(x)
    x = F.relu(model.fc2(x))
    x = model.dropout2(x)
    logits = model.fc3(x)
    print(f"Output shape (logits): \t{x.shape}") # Corrected variable name

    return logits

# Run the forward pass (output is ignored with _)
print("Running shape verification pass:")
_ = forward_with_shape_printing(model, dummy_input)