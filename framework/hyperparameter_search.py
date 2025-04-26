import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from framework.train import train_model
from framework.preprocess import load_data, preprocess_data, train_val_split
from framework.evaluation import classification_report

class HyperparameterSearch:
    def __init__(self, param_grid, epochs=20, batch_size=32, components_or_variance=None, df=None):
        """
        Initializes the hyperparameter search.

        Args:
            param_grid (dict): Dictionary where each key is a hyperparameter name (e.g., 'learning_rate')
                               and each value is a list of values to be tested.
            epochs (int): Default number of training epochs.
            batch_size (int): Default batch size.
            components_or_variance (int, float, or None): PCA dimensionality reduction parameter.
            df: DataFrame containing the dataset.
        """
        self.param_grid = param_grid
        self.default_epochs = epochs
        self.default_batch_size = batch_size
        self.components_or_variance = components_or_variance
        self.results = []

        X, y, pca_params = preprocess_data(df, components_or_variance=self.components_or_variance)
        self.X_train, self.y_train, self.X_val, self.y_val = train_val_split(X, y)
        self.pca_params = pca_params

    def search(self):
        """
        Perform a grid search over the hyperparameters, training a model for each combination.
        """
        keys, values = zip(*self.param_grid.items())

        for combination in itertools.product(*values):
            hyperparams = dict(zip(keys, combination))
            print(f"Testing parameters: {hyperparams}")

            # Resolve defaults
            epochs = hyperparams.get('epochs', self.default_epochs)
            batch_size = hyperparams.get('batch_size', self.default_batch_size)
            loss_function = hyperparams.get('loss_function', "cross_entropy")
            n_layers = hyperparams.get('n_layers', 1)

            # Train model
            trained_model, history = train_model(
                self.X_train, self.y_train, self.X_val, self.y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=hyperparams['learning_rate'],
                hidden_dim=hyperparams['hidden_dim'],
                reg_lambda=hyperparams['reg_lambda'],
                n_layers=n_layers,
                loss_function=loss_function,
                pca_params=self.pca_params
            )

            # Extract final metrics
            final_val_acc = history['val_acc'][-1] if 'val_acc' in history else None
            final_train_acc = history['train_acc'][-1] if 'train_acc' in history else None
            final_train_loss = history['loss'][-1] if 'loss' in history else None

            # Generate classification report on validation set
            val_preds = trained_model.predict(self.X_val)
            eval_metrics = classification_report(
                y_true=self.y_val,
                y_pred=val_preds,
                label_mapping=None,
                visualize_cm=False
            )

            # Determine hidden layers from hyperparams.
            if isinstance(hyperparams['hidden_dim'], list):
                hidden_layers = hyperparams['hidden_dim']
            else:
                hidden_layers = [hyperparams['hidden_dim']]
                current_dim = hyperparams['hidden_dim']
                for _ in range(1, n_layers):
                    current_dim = max(current_dim // 2, 1)
                    hidden_layers.append(current_dim)

            # Store results (including the full history)
            self.results.append({
                'n_layers': n_layers,
                'hidden_layers': hidden_layers,
                'learning_rate': hyperparams['learning_rate'],
                'n_epochs': epochs,
                'loss_function': loss_function,
                'reg_lambda': hyperparams['reg_lambda'],
                'batch_size': hyperparams.get('batch_size', self.default_batch_size),
                'Score': eval_metrics.get('score', None),
                'Accuracy': eval_metrics.get('accuracy', None),
                'F1': eval_metrics.get('f1', None),
                'Precision': eval_metrics.get('precision', None),
                'Recall': eval_metrics.get('recall', None),
                'train_accuracy': final_train_acc,
                'train_loss': final_train_loss,
                'history': history
            })

            print(f"Finished: {hyperparams} -> Final Val Accuracy: {final_val_acc:.4f}\n")
        
        # Save results to file
        self.save_results()

    def save_results(self, filename="hyperparameter_results.csv"):
        """
        Saves the results to a CSV file.
        """
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def load_results(self, filename="hyperparameter_results.csv"):
        """
        Loads saved hyperparameter search results.
        """
        df = pd.read_csv(filename)
        print(df.head())
        return df

    def get_best(self):
        """
        Return the best hyperparameter combination based on validation accuracy.
        """
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        df = df.dropna(subset=['Accuracy'])
        if df.empty:
            print("No valid results found.")
            return None

        best_idx = df['Accuracy'].astype(float).idxmax()
        return df.iloc[int(best_idx)]


    def plot_results(self):
        """
        Visualizes the final validation accuracy for each hyperparameter configuration using a bar plot.
        """
        if not self.results:
            print("No results to plot.")
            return

        labels = []
        accuracies = []
        for result in self.results:
            label = (f"lr:{result['learning_rate']}, hd:{result['hidden_layers']}, reg:{result['reg_lambda']}, "
                     f"nl:{result['n_layers']}, bs:{result.get('batch_size', self.default_batch_size)}, "
                     f"ep:{result.get('n_epochs', self.default_epochs)}")
            labels.append(label)
            accuracies.append(result['Accuracy'])

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(accuracies)), accuracies)
        plt.xticks(range(len(accuracies)), labels, rotation=45, ha="right")
        plt.ylabel("Final Validation Accuracy")
        plt.title("Hyperparameter Search Results")
        plt.tight_layout()
        plt.show()

    def plot_best_history(self):
        """
        Plots the learning curves for the best hyperparameter configuration.
        Displays Loss and Accuracy over epochs.
        """
        best = self.get_best()
        if best is None:
            print("No best configuration found.")
            return

        history = best.get('history')
        if history is None:
            print("No history available for the best configuration.")
            return

        epochs_range = range(1, len(history['loss']) + 1)
        plt.figure(figsize=(12, 5))

        # Loss Curve.
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['loss'], marker='o', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # Accuracy Curve.
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], marker='o', label='Train Acc')
        plt.plot(epochs_range, history['val_acc'], marker='o', label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()

        plt.suptitle("Learning Curves for Best Configuration", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
