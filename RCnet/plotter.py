"""
This module provides the `Plotter` class for tracking and visualizing training loss.

The `Plotter` class allows users to:
- Track the loss history over training epochs.
- Visualize the loss progression in real-time using matplotlib.
- Save the loss plot to a specified file for later analysis.

Modules:
- os: Used for handling file and directory operations.
- typing.Optional: Allows specifying optional arguments for methods.
- matplotlib.pyplot: Used for plotting graphs and visualizations.

Classes:
    Plotter: A class for managing loss history and generating loss plots.

Usage Example:
    plotter = Plotter()
    plotter.update_loss(0.5)
    plotter.plot_loss()
    plotter.save_plot("training_loss.png")
"""

import os
from typing import Optional

import matplotlib.pyplot as plt


class Plotter:
    """
    A class for tracking and visualizing the loss during training.

    Attributes:
        loss_history (list): A list to store the history of loss values.

    Methods:
        update_loss(loss: float) -> None:
            Adds a new loss value to the loss history.

        plot_loss() -> None:
            Plots the loss history against epochs in real-time.

        save_plot(file_path: Optional[str] = "loss_plot.png") -> None:
            Saves the loss plot to the specified file path. Creates any necessary directories.
    """

    def __init__(self) -> None:
        """
        Initializes the Plotter with an empty loss history.
        """
        self.loss_history: list[float] = []

    def update_loss(self, loss: float) -> None:
        """
        Updates the loss history with a new loss value.

        Args:
            loss (float): The current loss value to be added to the history.
        """
        self.loss_history.append(loss)

    def plot_loss(self) -> None:
        """
        Plots the training loss history against epochs in real-time.

        The plot is updated during training to visualize the progress of the loss.
        """
        plt.clf()  # Clear the current figure
        plt.plot(self.loss_history, label="Training Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epoch")
        plt.legend()
        plt.draw()
        plt.pause(0.001)

    def save_plot(self, file_path: Optional[str] = "loss_plot.png") -> None:
        """
        Saves the training loss plot to the specified file path.

        Args:
            file_path (str, optional): The path where the plot will be saved. Defaults to "loss_plot.png".
        """
        plt.clf()  # Clear the current figure
        plt.plot(self.loss_history, label="Training Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epoch")
        plt.legend()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        print(f"Loss plot saved at {file_path}")
