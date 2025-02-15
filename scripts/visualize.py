import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_tsne(features, labels, num_classes=10,
              output_dir = "../experiments/", 
              experiment_name = "simclr/cifar10/",
              filename = "tsne_epochno.png"):
    """
    Plot t-SNE visualization of the given features
    Args:
        features: np.array of shape (N, D)
        labels: np.array of shape (N,)
        num_classes: int, number of classes in the dataset
    """
    
    # cgeck features/labels data type
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # perform t-SNE
    tsne = TSNE(n_components=2, random_state=0, max_iter=3000, perplexity=30)
    features = tsne.fit_transform(features)

    # plot t-SNE
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=features[:,0], y=features[:,1],
        hue=labels,
        palette=sns.color_palette("hsv", num_classes),
        legend="full",
        alpha=0.6
    )
    plt.title("t-SNE visualization")
    plt.savefig(output_dir + experiment_name + "visualizations/" + filename)
    plt.show()

def line_plot(x, y, x_label, y_label, title, output_dir, experiment_name, filename):
    """
    Line plot for given x and y
    Args:
        x: list, x-axis values
        y: list, y-axis values
        x_label: str, x-axis label
        y_label: str, y-axis label
        title: str, plot title
        output_dir: str, output directory
        experiment_name: str, experiment name
        filename: str, output filename
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.savefig(output_dir + experiment_name + "visualizations/" + filename)
    plt.show()