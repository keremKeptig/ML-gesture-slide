import numpy as np
import matplotlib.pyplot as plt

def compute_precision_recall_f1(y_true, y_pred):
    """
    Compute weighted precision, recall, and F1 score for multi-class classification.
    
    Args:
        y_true: Array-like of true labels.
        y_pred: Array-like of predicted labels.
    
    Returns:
        A tuple (weighted_precision, weighted_recall, weighted_f1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Identify unique labels
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    
    total_samples = len(y_true)
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for label in unique_labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        support = np.sum(y_true == label)
        
        total_precision += precision * support
        total_recall += recall * support
        total_f1 += f1 * support
    
    weighted_precision = total_precision / total_samples
    weighted_recall = total_recall / total_samples
    weighted_f1 = total_f1 / total_samples
    
    return weighted_precision, weighted_recall, weighted_f1

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    
    Args:
        y_true: Array-like of true labels.
        y_pred: Array-like of predicted labels.
        
    Returns:
        A tuple (cm, labels) where:
            cm: 2D numpy array (rows: true labels, columns: predicted labels)
            labels: Sorted array of unique labels corresponding to the rows/columns.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get sorted unique labels
    labels = np.sort(np.unique(np.concatenate((y_true, y_pred))))
    num_labels = labels.shape[0]
    
    cm = np.zeros((num_labels, num_labels), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    
    for true, pred in zip(y_true, y_pred):
        cm[label_to_index[true], label_to_index[pred]] += 1
        
    return cm, labels

def accuracy(y_true, y_pred):
    """
    Compute the overall accuracy.
    
    Args:
        y_true: Array-like of true labels.
        y_pred: Array-like of predicted labels.
        
    Returns:
        The accuracy as a float.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """
    Visualize the confusion matrix using Matplotlib.
    
    Args:
        cm: Confusion matrix (2D numpy array).
        labels: List or array of label names.
        title: Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def classification_report(y_true, y_pred, label_mapping=None, visualize_cm=False, title="Confusion Matrix"):
    """
    Compute and print a classification report and optionally visualize the confusion matrix.

    Args:
        y_true: Array-like of true labels.
        y_pred: Array-like of predicted labels.
        label_mapping: Dictionary mapping numeric labels to gesture names. If provided, 
                       the confusion matrix and report will use these names.
        visualize_cm: If True, displays the confusion matrix using plot_confusion_matrix.
        title: Title for the confusion matrix plot.
    
    Returns:
        report: A dictionary with keys 'accuracy', 'precision', 'recall', 'f1',
                'confusion_matrix', and 'labels'.
    """
    acc = accuracy(y_true, y_pred)
    precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred)
    cm, labels = confusion_matrix(y_true, y_pred)
    
    # Map numeric labels to gesture names if a mapping is provided.
    if label_mapping is not None:
        labels_named = [label_mapping.get(label, str(label)) for label in labels]
    else:
        labels_named = labels.tolist()
    report = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "labels": labels_named
    }
    
    print("Classification Report:")
    print("----------------------")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    if visualize_cm:
        plot_confusion_matrix(cm, labels_named, title=title)
    
    return report