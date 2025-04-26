import os
import numpy as np
import pandas as pd
from framework.pca import apply_pca

def load_data(data_dir=None):
    """
    Load all annotated CSV files from the specified directory and combine them into one DataFrame.
    Assumes each CSV has a "timestamp" column and a "ground_truth" column.
    """
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)  # Do NOT set index_col; keep timestamp as a column.
        if "ground_truth" in df.columns:
            dfs.append(df)
        else:
            print(f"Skipping {file} because it has no 'ground_truth' column.")
    if len(dfs) == 0:
        raise ValueError("No annotated CSV files found in " + data_dir)
    combined_df = pd.concat(dfs, axis=0)
    combined_df.sort_values(by="timestamp", inplace=True)
    return combined_df



def preprocess_data(df, components_or_variance=None, gesture_map=None):
    """
    Preprocess the combined annotated DataFrame by:
      1. Mapping gesture labels to integer values.
      2. Extracting feature columns (using absolute landmark coordinates) and converting them to a NumPy array.
      3. Optionally applying PCA to reduce dimensionality.

    We assume the CSV contains:
      - A "timestamp" column.
      - Landmark coordinate columns (e.g., "nose_x", "nose_y", "nose_z", etc.).
      - A "ground_truth" column.
      
    The final feature vector excludes unnecessary columns like timestamp.

    Args:
        df: DataFrame containing the raw CSV data.
        n_components: Number of principal components to keep (or a float threshold). If None, PCA is not applied.
    
    Returns:
        X: Preprocessed feature array.
        y: Gesture labels as integers.
        pca_params: A tuple (mean_X, components) if PCA is applied, otherwise None.
    """
    # Map gesture labels if we do not have default mandatory.
    if gesture_map is None:
        gesture_map = {
            "swipe_left": 0, 
            "swipe_right": 1, 
            "rotate": 2, 
            "idle": 3
        }   
    df['gesture_mapped'] = df["ground_truth"].astype(str).str.strip().str.lower().map(gesture_map)
    num_unmapped = df['gesture_mapped'].isnull().sum()
    if num_unmapped > 0:
        print(f"Info: {num_unmapped} rows did not match expected gestures; assigning them as 'idle'.")
        df.loc[df['gesture_mapped'].isnull(), 'gesture_mapped'] = gesture_map["idle"]
    df['gesture_mapped'] = df['gesture_mapped'].astype(np.int32)
    
    # Extract feature columns.
    # Exclude "ground_truth", "gesture_mapped", and "timestamp" so that only the absolute landmark coordinates are used.
    feature_columns = [col for col in df.columns if col not in ["ground_truth", "gesture_mapped", "timestamp"]]
    print("Feature columns used for training:", feature_columns)
    X = df[feature_columns].values.astype(np.float32)
    y = df["gesture_mapped"].values.astype(np.int32)
        
    pca_params = None
    if components_or_variance is not None:
        X_pca, mean_X, components = apply_pca(X, components_or_variance)
        if mean_X is not None and components is not None:
            X = X_pca
            pca_params = (mean_X, components)
    
    return X, y, pca_params

def train_val_split(X, y, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(train_ratio * len(indices))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    return X_train, y_train, X_val, y_val


