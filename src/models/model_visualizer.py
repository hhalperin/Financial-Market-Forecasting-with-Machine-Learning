import os
import matplotlib.pyplot as plt
import json
import numpy as np

class ModelVisualizer:
    def __init__(self, metrics_file='training_metrics.json', figures_dir='figures'):
        self.metrics_file = metrics_file
        self.metrics = {'train_loss': [], 'val_loss': []}
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)

    def _save_figure(self, save_path, title):
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")

    def log_metrics(self, train_loss, val_loss):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f)

    def plot_learning_curve(self, model_name='model', hyperparameters="default"):
        sub_dir = os.path.join(self.figures_dir, f"{model_name}_{hyperparameters.replace(' ', '_')}")
        os.makedirs(sub_dir, exist_ok=True)

        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            print("Metrics file not found. Please ensure that training is logged.")
            return

        epochs = range(1, len(self.metrics['train_loss']) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_loss'], label='Training Loss', color='blue', marker='o')
        plt.plot(epochs, self.metrics['val_loss'], label='Validation Loss', color='red', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        save_path = os.path.join(sub_dir, f"{model_name}_learning_curve.png")
        self._save_figure(save_path, "Learning Curve: Training and Validation Loss")

    def plot_actual_vs_predicted(self, y_true, y_pred, model_name='model', hyperparameters="default"):
        sub_dir = os.path.join(self.figures_dir, f"{model_name}_{hyperparameters.replace(' ', '_')}")
        os.makedirs(sub_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label='Actual', color='blue', alpha=0.7, linestyle='-', marker='.')
        plt.plot(y_pred, label='Predicted', color='red', alpha=0.7, linestyle='-', marker='.')
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.legend()

        save_path = os.path.join(sub_dir, f"{model_name}_actual_vs_predicted.png")
        self._save_figure(save_path, "Actual vs. Predicted Values")
