import os 
import sys
import numpy as np
from framework.model import NeuralNetwork
from framework.pca import save_pca  # Use the new save_pca function from pca.py

def save_model(nn, model_path="models/model.npz"):
    """
    Save the model parameters to the specified path using NumPy's function.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    params = {}
    for i, (w, b) in enumerate(zip(nn.weights, nn.biases)):
        params[f'W{i+1}'] = w
        params[f'b{i+1}'] = b
    np.savez(model_path, **params)
    print(f"Model saved to {model_path}")

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=0.01, 
                hidden_dim=[64,32], n_layers=3, reg_lambda=0.01, loss_function="cross_entropy", 
                pca_params=None):
    """
    Train the neural network using pre-split training and validation data.
    
    Args:
        X_train: Training feature array.
        y_train: Training labels.
        X_val: Validation feature array.
        y_val: Validation labels.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for gradient descent.
        hidden_dim: Number of neurons per hidden layer.
        n_layers: Number of hidden layers.
        reg_lambda: L2 regularization strength.
        loss_function: Loss function to use ("cross_entropy" or "mse").
        pca_params: Tuple (mean_X, components) if PCA was applied; otherwise, None.
    
    Returns:
        nn: Trained neural network model.
        history: Dictionary containing loss, training accuracy, and validation accuracy history.
    """
    # If PCA was applied externally, save the PCA parameters.
    if pca_params is not None:
        os.makedirs("models", exist_ok=True)
        save_pca(pca_params[0], pca_params[1], "models/pca_params.pkl")
        print("PCA parameters saved to models/pca_params.pkl")
    
    input_dim = X_train.shape[1]  
    output_dim = len(np.unique(y_train))  

    # Initialize the neural network.
    nn = NeuralNetwork(input_dim, hidden_dim, output_dim, n_layers=n_layers,
                       learning_rate=learning_rate, reg_lambda=reg_lambda)
    
    history = {
        'loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Variables for tracking the best model based on validation accuracy.
    best_val_acc = 0.0
    best_model_weights = None
    best_model_biases = None

    for epoch in range(epochs):
        # Shuffle the training data.
        perm = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        epoch_loss = 0
        m = 0
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            m += X_batch.shape[0]
            
            outputs = nn.forward(X_batch)
            if loss_function == "cross_entropy":
                loss = nn.compute_loss(outputs, y_batch)
            elif loss_function == "mse":
                loss = nn.compute_mse_loss(outputs, y_batch)
            else:
                raise ValueError("Invalid loss function. Use 'cross_entropy' or 'mse'.")
            
            epoch_loss += loss
            nn.backward(X_batch, y_batch)
        
        epoch_loss /= (m / batch_size)
        
        # Compute predictions on training and validation sets.
        train_preds = nn.predict(X_train)
        val_preds = nn.predict(X_val)
        
        # Calculate accuracies.
        train_acc = np.mean(train_preds == y_train)
        val_acc = np.mean(val_preds == y_val)
        
        history['loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = [w.copy() for w in nn.weights]
            best_model_biases = [b.copy() for b in nn.biases]
    
    # Restore the best model parameters.
    if best_model_weights is not None:
        for i in range(len(nn.weights)):
            nn.weights[i] = best_model_weights[i]
            nn.biases[i] = best_model_biases[i]
        print(f"Best model selected with validation accuracy: {best_val_acc:.4f}")
        save_model(nn, model_path="models/best_model.npz")
    
    # Save the final model.
    save_model(nn)
    return nn, history
