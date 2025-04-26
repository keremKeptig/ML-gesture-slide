import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, learning_rate=0.01, reg_lambda=0.0):
        """
        Initialize a Neural Network.
        If hidden_dim is a list, it is used directly; otherwise, a list of hidden layer sizes is generated based on n_layers.
        
        Parameters:
            input_dim: Number of features per input.
            hidden_dim: Either an integer (size of the first hidden layer) or a list of integers for hidden layers.
            output_dim: Number of output classes.
            n_layers: Number of hidden layers (used only if hidden_dim is an integer; if hidden_dim is a list, its length is used).
            learning_rate: Learning rate for gradient descent.
            reg_lambda: L2 regularization strength.
        """
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.weights = []
        self.biases = []
        
        # Determine the hidden layer sizes.
        if isinstance(hidden_dim, list):
            hidden_layers = hidden_dim
            if len(hidden_layers) != n_layers:
                print("Warning: The length of hidden_dim list does not match n_layers. Using the list's length.")
                n_layers = len(hidden_layers)
        else:
            hidden_layers = [hidden_dim]
            current_dim = hidden_dim
            for _ in range(1, n_layers):
                current_dim = max(current_dim // 2, 1)
                hidden_layers.append(current_dim)
        
        # Save the number of hidden layers.
        self.n_layers = n_layers
        
        # Build the network: input layer to first hidden layer.
        self.weights.append(np.random.randn(input_dim, hidden_layers[0]) * np.sqrt(2.0 / input_dim))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        
        # Build additional hidden layers.
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]) * np.sqrt(2.0 / hidden_layers[i-1]))
            self.biases.append(np.zeros((1, hidden_layers[i])))
        
        # Build the output layer.
        self.weights.append(np.random.randn(hidden_layers[-1], output_dim) * np.sqrt(2.0 / hidden_layers[-1]))
        self.biases.append(np.zeros((1, output_dim)))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        # Numerically stable softmax.
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Perform a forward pass through the network.
        Stores intermediate linear combinations (z) and activations (a) for backpropagation.
        """
        self.layer_inputs = []      # Store z values for each layer.
        self.layer_activations = [X]  # Start with input
        
        A = X
        # Forward pass through each hidden layer.
        for i in range(self.n_layers):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(Z)
            A = self.relu(Z)
            self.layer_activations.append(A)
        
        # Output layer.
        Z_out = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(Z_out)
        output = self.softmax(Z_out)
        self.layer_activations.append(output)
        return output
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss with L2 regularization.
        """
        m = y_true.shape[0]
        one_hot = np.zeros_like(y_pred)
        one_hot[np.arange(m), y_true] = 1
        cross_entropy_loss = -np.sum(one_hot * np.log(y_pred + 1e-8)) / m
        
        # L2 regularization loss
        reg_loss = 0
        for w in self.weights:
            reg_loss += 0.5 * self.reg_lambda * np.sum(w * w)
        return cross_entropy_loss + reg_loss
    
    def compute_mse_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error (MSE) loss between the predicted probabilities and one-hot encoded true labels.
        
        Args:
            y_pred: Predicted probabilities from softmax, shape (n_samples, n_classes).
            y_true: True class labels as integers, shape (n_samples,).
        
        Returns:
            mse_loss: Mean Squared Error loss (scalar).
        """
        m = y_true.shape[0]
        one_hot = np.zeros_like(y_pred)
        one_hot[np.arange(m), y_true] = 1
        mse_loss = np.sum((y_pred - one_hot) ** 2) / m
        return mse_loss
    
    def backward(self, X, y_true):
        """
        Perform backpropagation to update weights and biases.
        """
        m = y_true.shape[0]
        one_hot = np.zeros_like(self.layer_activations[-1])
        one_hot[np.arange(m), y_true] = 1
        
        # Gradient at the output layer.
        dZ = (self.layer_activations[-1] - one_hot) / m
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        
        # Gradients for output layer.
        dW = np.dot(self.layer_activations[-2].T, dZ) + self.reg_lambda * self.weights[-1]
        db = np.sum(dZ, axis=0, keepdims=True)
        grad_weights[-1] = dW
        grad_biases[-1] = db
        
        # Backpropagate through hidden layers.
        dA_prev = np.dot(dZ, self.weights[-1].T)
        for i in range(self.n_layers - 1, -1, -1):
            dZ = dA_prev * self.relu_derivative(self.layer_inputs[i])
            dW = np.dot(self.layer_activations[i].T, dZ) + self.reg_lambda * self.weights[i]
            db = np.sum(dZ, axis=0, keepdims=True)
            grad_weights[i] = dW
            grad_biases[i] = db
            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
        
        # Update weights and biases.
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]
    
    def predict(self, X):
        """
        Predict class labels for input samples.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
