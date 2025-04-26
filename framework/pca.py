from matplotlib import pyplot as plt
import numpy as np
import pickle

def apply_pca(X, components_or_variance=None):
    """
    Apply PCA to reduce the dimensionality of X using only NumPy.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        components_or_variance: either an integer representing the number of principal components 
                                to keep, or a float between 0 and 1 representing the minimum explained 
                                variance ratio to be preserved.
    
    Returns:
        X_pca: Transformed data of shape (n_samples, n_components)
        mean_X: Mean of the original data
        components: The top principal components (eigenvectors)
    """
    # Center the data
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X

    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigen decomposition (using np.linalg.eigh for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Determine the number of components based on components_or_variance
    if isinstance(components_or_variance, float) and 0 < components_or_variance < 1:
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components_int = np.searchsorted(cumulative_variance, components_or_variance) + 1
        n_components = n_components_int
    elif isinstance(components_or_variance, int) and components_or_variance > 0:
        n_components = components_or_variance
    elif components_or_variance is None:
        # If None is passed, skip PCA.
        return X, None, None
    else:
        raise ValueError("components_or_variance must be a positive integer or a float between 0 and 1 representing variance threshold")
    
    # Select the top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    
    # Project the data
    X_pca = np.dot(X_centered, components)
    
    return X_pca, mean_X, components

def save_pca(mean_X, components, filepath):
    """
    Save PCA parameters (mean and components) to a file using pickle.
    
    Args:
        mean_X: Mean of the original data computed during PCA.
        components: Principal components (eigenvectors).
        filepath: Path to the file where the PCA parameters should be saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump((mean_X, components), f)
    print(f"PCA parameters saved to {filepath}")

def load_pca(filepath):
    """
    Load PCA parameters from a file.
    
    Args:
        filepath: Path to the file where PCA parameters were saved.
        
    Returns:
        A tuple (mean_X, components) that were saved using save_pca.
    """
    with open(filepath, 'rb') as f:
        mean_X, components = pickle.load(f)
    print(f"PCA parameters loaded from {filepath}")
    return mean_X, components



def apply_existing_pca(X, pca_params):
    """
    Apply an existing PCA transformation using given PCA parameters.
    
    Args:
        X: The raw feature array.
        pca_params: A tuple (mean_X, components) obtained from training data.
        
    Returns:
        X_pca: The PCA-transformed feature array.
    """
    if pca_params is None:
        return X
    mean_X, components = pca_params
    X_centered = X - mean_X
    return np.dot(X_centered, components)

def visualize_pca_explained_variance(X, components_or_variance=0.99):
    """
    Apply PCA and visualize how much variance each component explains.

    Args:
        X (numpy.ndarray): The original high-dimensional feature matrix.
        components_or_variance (float or int): PCA configuration.
    """
    # Apply PCA
    _, mean_X, components = apply_pca(X, components_or_variance)
    
    # Compute explained variance ratio
    eigenvalues = np.var(np.dot(X - mean_X, components), axis=0)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Plot explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid()
    plt.show()
