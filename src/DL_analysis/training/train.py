# train.py
import torch
import matplotlib.pyplot as plt


def train(model, train_loader, criterion, optimizer, device):
    """
        Perform one epoch of training.

        - Sets the model in training mode.
        - Iterates through training data batches.
        - Computes the forward pass and loss.
        - Performs backpropagation and optimizer step.
        - Accumulates total training loss to return the average at the end.
    """
    # Set the model to training mode
    model.train()
    # Track cumulative loss
    running_loss = 0.0
    correct_train = 0

    for batch in train_loader:
        x_train, y_train = batch['X'].to(device), batch['y'].to(device)

        # Reset gradients to zero before each step
        optimizer.zero_grad()

        # Forward pass: compute predictions
        outputs = model(x_train)
        # Compute loss between predictions and labels
        loss = criterion(outputs, y_train)
        # Backpropagation: compute gradients
        loss.backward()
        # Update model weights using gradients
        optimizer.step()

        # Accumulate the loss weighted by batch size
        running_loss += loss.item() * x_train.size(0)
        
        # Calculate accuracy for this batch
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == y_train).sum().item()

    # Compute average loss for the epoch
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct_train / len(train_loader.dataset)
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    """
        Evaluate the model on the validation set.

        - Sets the model in evaluation mode.
        - Disables gradient computation.
        - Computes forward pass and accumulates loss and accuracy.
        - Returns average loss and overall accuracy on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0  # Count of correct predictions

    # Disable gradient computation for memory/speed
    with torch.no_grad():
        for batch in val_loader:
            x_val, y_val = batch['X'].to(device), batch['y'].to(device)

            # Forward pass
            outputs = model(x_val)
            # Compute loss
            loss = criterion(outputs, y_val)
            # Accumulate loss
            running_loss += loss.item() * x_val.size(0)

            # Get predicted class indices
            _, predicted = torch.max(outputs, 1)

            # Count correct predictions
            correct += (predicted == y_val).sum().item()


    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)  # Compute accuracy

    return val_loss, val_accuracy

def plot_training_curves(history, save_dir):
    """
    Plot and save training curves for Loss and Accuracy.
    history: dict containing 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists.
    save_dir: directory where to save the plots.
    """
    import os
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Prediction: Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='tab:blue')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='s', color='tab:orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
    plt.close()

    # 2. Prediction: Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o', color='tab:green')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='s', color='tab:red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300)
    plt.close()