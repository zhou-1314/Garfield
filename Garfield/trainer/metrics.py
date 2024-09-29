"""
This module contains metrics to evaluate the Garfield model training.
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import torch
from matplotlib.ticker import MaxNLocator


def eval_metrics(
        edge_recon_probs: Union[torch.Tensor, np.ndarray],
        edge_labels: Union[torch.Tensor, np.ndarray],
        omics_recon_pred: Union[torch.Tensor, np.ndarray]=None,
        omics_recon_truth: Union[torch.Tensor, np.ndarray]=None) -> dict:
    """
    Get the evaluation metrics for a (balanced) sample of positive and negative
    edges and a sample of nodes.

    Parameters
    ----------
    edge_recon_probs:
        Tensor or array containing reconstructed edge probabilities.
    edge_labels:
        Tensor or array containing ground truth labels of edges.

    Returns
    ----------
    eval_dict:
        Dictionary containing the evaluation metrics ´auroc_score´ (area under
        the  receiver operating characteristic curve), ´auprc score´ (area under
        the precision-recall curve), ´best_acc_score´ (accuracy under optimal
        classification threshold) and ´best_f1_score´ (F1 score under optimal
        classification threshold).
    """
    eval_dict = {}

    if isinstance(edge_recon_probs, torch.Tensor):
        edge_recon_probs = edge_recon_probs.detach().cpu().numpy()
    if isinstance(edge_labels, torch.Tensor):
        edge_labels = edge_labels.detach().cpu().numpy()
    if isinstance(omics_recon_pred, torch.Tensor):
        omics_recon_pred = omics_recon_pred.detach().cpu().numpy()
    if isinstance(omics_recon_truth, torch.Tensor):
        omics_recon_truth = omics_recon_truth.detach().cpu().numpy()

    if omics_recon_pred is not None and omics_recon_truth is not None:
        # Calculate the gene expression mean squared error
        eval_dict["gene_expr_mse_score"] = skm.mean_squared_error(
            omics_recon_truth,
            omics_recon_pred)

    # Calculate threshold independent metrics
    eval_dict["auroc_score"] = skm.roc_auc_score(edge_labels, edge_recon_probs)
    eval_dict["auprc_score"] = skm.average_precision_score(edge_labels,
                                                           edge_recon_probs)

    # Get the optimal classification probability threshold above which an edge
    # is classified as positive so that the threshold optimizes the accuracy
    # over the sampled (balanced) set of positive and negative edges.
    best_acc_score = 0
    best_threshold = 0
    for threshold in np.arange(0.01, 1, 0.005):
        pred_labels = (edge_recon_probs > threshold).astype("int")
        acc_score = skm.accuracy_score(edge_labels, pred_labels)
        if acc_score > best_acc_score:
            best_threshold = threshold
            best_acc_score = acc_score
    eval_dict["best_acc_score"] = best_acc_score
    eval_dict["best_acc_threshold"] = best_threshold

    # Get the optimal classification probability threshold above which an edge
    # is classified as positive so that the threshold optimizes the F1 score
    # over the sampled (balanced) set of positive and negative edges.
    best_f1_score = 0
    for threshold in np.arange(0.01, 1, 0.005):
        pred_labels = (edge_recon_probs > threshold).astype("int")
        f1_score = skm.f1_score(edge_labels, pred_labels)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
    eval_dict["best_f1_score"] = best_f1_score
    return eval_dict


def plot_eval_metrics(eval_dict: dict) -> plt.figure:
    """
    Plot evaluation metrics.

    Parameters
    ----------
    eval_dict:
        Dictionary containing the eval metric scores to be plotted.

    Returns
    ----------
    fig:
        Matplotlib figure containing a plot of the evaluation metrics.
    """
    # Plot epochs as integers
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot eval metrics
    for metric_key, metric_scores in eval_dict.items():
        plt.plot(metric_scores, label=metric_key)
    plt.title("Evaluation metrics over epochs")
    plt.ylabel("metric score")
    plt.xlabel("epoch")
    plt.legend(loc="lower right")

    # Retrieve figure
    fig = plt.gcf()
    plt.close()
    return fig