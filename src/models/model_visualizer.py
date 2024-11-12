import os
import matplotlib.pyplot as plt
import json

class ModelVisualizer:
    """
    A utility class to visualize model training and validation metrics.
    """
    
    def __init__(self, metrics_file='training_metrics.json', figures_dir='figures'):
        self.metrics_file = metrics_file
        self.metrics = {'train_loss': [], 'val_loss': []}
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)

    def log_metrics(self, train_loss, val_loss):
        """
        Logs training and validation loss.

        Parameters:
            train_loss (float): The average training loss for the epoch.
            val_loss (float): The average validation loss for the epoch.
        """
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        # Save metrics to a JSON file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f)

    def plot_metrics(self, model_name='model'):
        """
        Plots the training and validation loss over epochs and saves the figure.

        Parameters:
            model_name (str): The name of the model used in the filename for the saved figure.
        """
        # Load metrics from the JSON file if available
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            print("Metrics file not found. Please ensure that training is logged.")
            return

        # Plot the metrics
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_loss'], label='Training Loss', color='blue', marker='o')
        plt.plot(epochs, self.metrics['val_loss'], label='Validation Loss', color='red', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)

        # Save figure
        figure_filename = os.path.join(self.figures_dir, f"{model_name}_training_validation_loss.png")
        plt.savefig(figure_filename)
        print(f"Training and validation loss plot saved to {figure_filename}")

        plt.close()  # Close the figure to avoid pausing

if __name__ == "__main__":
    visualizer = ModelVisualizer()
    visualizer.plot_metrics(model_name='example_model')
