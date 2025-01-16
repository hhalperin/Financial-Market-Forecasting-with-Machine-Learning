# models/model_visualizer.py

import io
import matplotlib.pyplot as plt

class ModelVisualizer:
    def __init__(self, data_handler, model_stage="models", model_name="model"):
        """
        :param data_handler: for saving plots
        :param model_stage: subfolder to store them
        :param model_name: prefix for file naming
        """
        self.data_handler = data_handler
        self.model_stage = model_stage
        self.model_name = model_name

    def plot_learning_curve(self, training_history):
        plt.figure(figsize=(10,6))
        epochs = range(1, len(training_history['train_loss'])+1)
        plt.plot(epochs, training_history['train_loss'], label='Train Loss')
        plt.plot(epochs, training_history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        fig_filename = f"{self.model_name}_learning_curve.png"
        self.save_figure(fig_filename)

    def plot_actual_vs_predicted(self, y_true, y_pred):
        plt.figure(figsize=(10,6))
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red')
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        fig_filename = f"{self.model_name}_actual_vs_predicted.png"
        self.save_figure(fig_filename)

    def plot_time_series(self, timestamps, y_true, y_pred):
        plt.figure(figsize=(10,6))
        plt.plot(timestamps, y_true, label='Actual', color='blue')
        plt.plot(timestamps, y_pred, label='Predicted', color='red')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        fig_filename = f"{self.model_name}_time_series.png"
        self.save_figure(fig_filename)

    def save_figure(self, fig_filename):
        # Let data_handler handle local vs S3
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        self.data_handler.save_figure_bytes(buffer.read(), fig_filename, stage=self.model_stage)
        plt.close()
