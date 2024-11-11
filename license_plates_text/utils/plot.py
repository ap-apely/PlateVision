import matplotlib.pyplot as plt
import numpy as np


def plot_losses(train_loss, valid_loss):
    """
    Plot training and validation losses over epochs.

    Parameters:
        train_losses (list): List of training loss values.
        valid_losses (list): List of validation loss values.

    Returns:
        None
    """
    plt.style.use("seaborn")
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_loss, color="blue", label="Training loss")
    ax.plot(valid_loss, color="red", label="Validation loss")
    ax.set(title="Loss over epochs", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    plt.style.use("default")
    _graph_name = "logs/losses.png"
    print(f"saving losses graph at {_graph_name}")
    plt.savefig(_graph_name)


def plot_acc(accuracy):
    """
    Plot model accuracy over epochs.

    Parameters:
        accuracy (list): List of accuracy values.

    Returns:
        None
    """
    plt.style.use("seaborn")
    accuracy = np.array(accuracy)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(accuracy, color="purple", label="Model Accuracy")
    ax.set(title="Accuracy over epochs", xlabel="Epoch", ylabel="Accuracy")
    ax.legend()
    plt.style.use("default")
    _graph_name = "logs/accuracy.png"
    print(f"saving accuracy graph at {_graph_name}")
    plt.savefig(_graph_name)
