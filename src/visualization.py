"""
visualization.py - Data visualization utilities for sentiment analysis

This module provides functions to create visualizations for sentiment analysis
results, model comparisons, and data distributions.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from typing import Dict, List, Optional, Tuple, Union, Any
import itertools
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set default plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# Color palettes
CATEGORY_COLORS = {
    'positive': '#2ecc71',  # Green
    'negative': '#e74c3c',  # Red
    'neutral': '#3498db',   # Blue
    'unknown': '#95a5a6'    # Gray
}

MODEL_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
]


def create_distribution_plot(
    labels: List[int],
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a bar plot showing the distribution of classes in the dataset.
    
    Args:
        labels: List of class labels (integers)
        class_names: List of class names corresponding to the labels
        title: Plot title
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Matplotlib Figure object
    """
    # Count occurrences of each class
    label_counts = pd.Series(labels).value_counts().sort_index()
    
    # Ensure all classes are represented
    all_labels = range(len(class_names))
    for label in all_labels:
        if label not in label_counts.index:
            label_counts[label] = 0
    
    label_counts = label_counts.sort_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(range(len(class_names)), label_counts, color=[CATEGORY_COLORS.get(name.lower(), 'gray') for name in class_names])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)} ({height/sum(label_counts)*100:.1f}%)',
                ha='center', va='bottom')
    
    # Customize plot
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_confusion_matrix_plot(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Create a confusion matrix visualization.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        class_names: List of class names
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
        cmap: Colormap for the confusion matrix
    
    Returns:
        Matplotlib Figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    
    # Customize plot
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_roc_curve_plot(
    y_true: List[int],
    y_scores: np.ndarray,
    class_names: List[str],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create ROC curve plots for multi-class classification.
    
    Args:
        y_true: List of true labels (integers)
        y_scores: Array of prediction scores (probabilities) with shape (n_samples, n_classes)
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert labels to one-hot encoding for micro-average
    y_true_bin = np.zeros((len(y_true), len(class_names)))
    for i, label in enumerate(y_true):
        y_true_bin[i, label] = 1
    
    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    # Plot micro-average ROC curve
    ax.plot(fpr_micro, tpr_micro, 
            label=f'Micro-average (AUC = {roc_auc_micro:.2f})',
            linestyle=':', linewidth=4)
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        # Binarize the data for this class
        y_true_bin = np.array([1 if label == i else 0 for label in y_true])
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true_bin, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve for this class
        ax.plot(fpr, tpr, 
                label=f'{class_name} (AUC = {roc_auc:.2f})',
                color=CATEGORY_COLORS.get(class_name.lower(), None))
    
    # Plot random guessing line
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Customize plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_precision_recall_plot(
    y_true: List[int],
    y_scores: np.ndarray,
    class_names: List[str],
    title: str = "Precision-Recall Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create precision-recall curve plots for multi-class classification.
    
    Args:
        y_true: List of true labels (integers)
        y_scores: Array of prediction scores (probabilities) with shape (n_samples, n_classes)
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve for each class
    for i, class_name in enumerate(class_names):
        # Binarize the data for this class
        y_true_bin = np.array([1 if label == i else 0 for label in y_true])
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_bin, y_scores[:, i])
        
        # Compute average precision
        ap = np.mean(precision)
        
        # Plot precision-recall curve for this class
        ax.plot(recall, precision, 
                label=f'{class_name} (AP = {ap:.2f})',
                color=CATEGORY_COLORS.get(class_name.lower(), None))
    
    # Customize plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="best")
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_model_comparison_plot(
    model_metrics: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a bar plot comparing multiple models across several metrics.
    
    Args:
        model_metrics: Dictionary mapping model names to dictionaries of metrics
        metrics_to_plot: List of metric names to include in the plot
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Matplotlib Figure object
    """
    # Create DataFrame from metrics
    data = []
    for model_name, metrics in model_metrics.items():
        model_data = {'Model': model_name}
        for metric in metrics_to_plot:
            if metric in metrics:
                model_data[metric] = metrics[metric]
        data.append(model_data)
    
    df = pd.DataFrame(data)
    df = df.set_index('Model')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grouped bar chart
    df[metrics_to_plot].plot(kind='bar', ax=ax)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    # Customize plot
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_ylim([0, 1.05])
    ax.legend(title='Metric')
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_radar_chart(
    model_metrics: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    title: str = "Model Comparison Radar Chart",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Create a radar chart comparing multiple models across several metrics.
    
    Args:
        model_metrics: Dictionary mapping model names to dictionaries of metrics
        metrics_to_plot: List of metric names to include in the chart
        title: Chart title
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Matplotlib Figure object
    """
    # Number of metrics to plot
    N = len(metrics_to_plot)
    
    # Create angles for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot each model
    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        # Extract values for each metric
        values = [metrics.get(metric, 0) for metric in metrics_to_plot]
        values += values[:1]  # Close the loop
        
        # Plot model on radar chart
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                color=MODEL_COLORS[i % len(MODEL_COLORS)])
        ax.fill(angles, values, alpha=0.1, color=MODEL_COLORS[i % len(MODEL_COLORS)])
    
    # Customize plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim([0, 1])
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_error_analysis_plot(
    texts: List[str],
    true_labels: List[int],
    pred_labels: List[int],
    confidences: List[float],
    class_names: List[str],
    title: str = "Error Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Create a comprehensive error analysis visualization.
    
    Args:
        texts: List of text samples
        true_labels: List of true labels
        pred_labels: List of predicted labels
        confidences: List of prediction confidences
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Matplotlib Figure object
    """
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'pred_label': pred_labels,
        'confidence': confidences,
        'correct': np.array(true_labels) == np.array(pred_labels)
    })
    
    # Add text length
    df['text_length'] = df['text'].apply(len)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    gs = fig.add_gridspec(2, 3)
    
    # 1. Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix')
    
    # 2. Confidence distribution by correctness
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=df, x='confidence', hue='correct', bins=20, multiple='dodge', ax=ax2)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution by Correctness')
    ax2.legend(title='Correct', labels=['Incorrect', 'Correct'])
    
    # 3. Error rate by text length
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Create bins for text length
    length_bins = [0, 50, 100, 150, 200, np.inf]
    length_bin_labels = ['0-50', '51-100', '101-150', '151-200', '200+']
    
    df['length_bin'] = pd.cut(df['text_length'], bins=length_bins, labels=length_bin_labels)
    
    # Calculate error rate for each bin
    error_rates = df.groupby('length_bin')['correct'].mean().map(lambda x: 1 - x) * 100
    counts = df.groupby('length_bin').size()
    
    sns.barplot(x=error_rates.index, y=error_rates.values, ax=ax3)
    ax3.set_xlabel('Text Length')
    ax3.set_ylabel('Error Rate (%)')
    ax3.set_title('Error Rate by Text Length')
    
    # Add count labels
    for i, (rate, count) in enumerate(zip(error_rates, counts)):
        ax3.text(i, rate + 1, f'n={count}', ha='center')
    
    # 4. Confusion matrix focusing on errors
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Filter for incorrect predictions
    errors = df[~df['correct']]
    
    if len(errors) > 0:
        error_cm = confusion_matrix(errors['true_label'], errors['pred_label'])
        sns.heatmap(error_cm, annot=True, fmt='d', cmap="Reds",
                    xticklabels=class_names, yticklabels=class_names, ax=ax4)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        ax4.set_title('Confusion Matrix of Errors')
    else:
        ax4.text(0.5, 0.5, 'No errors to display', ha='center', va='center')
        ax4.set_title('Confusion Matrix of Errors')
    
    # 5. Top misclassifications
    ax5 = fig.add_subplot(gs[1, 1:])
    
    if len(errors) > 0:
        # Get most confident misclassifications
        top_errors = errors.nlargest(5, 'confidence')
        
        # Format text for display
        error_texts = []
        for _, row in top_errors.iterrows():
            text = row['text']
            if len(text) > 50:
                text = text[:50] + '...'
            
            true_class = class_names[row['true_label']]
            pred_class = class_names[row['pred_label']]
            conf = row['confidence']
            
            error_texts.append(f"{text}\nTrue: {true_class}, Pred: {pred_class}, Conf: {conf:.2f}")
        
        if error_texts:
            ax5.axis('off')
            ax5.set_title('Most Confident Misclassifications')
            
            for i, txt in enumerate(error_texts):
                ax5.text(0, 1 - i*0.2, txt, va='top', wrap=True, fontsize=10)
        else:
            ax5.text(0.5, 0.5, 'No errors to display', ha='center', va='center')
            ax5.set_title('Most Confident Misclassifications')
    else:
        ax5.text(0.5, 0.5, 'No errors to display', ha='center', va='center')
        ax5.set_title('Most Confident Misclassifications')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_tsne_plot(
    embeddings: np.ndarray,
    labels: List[int],
    class_names: List[str],
    title: str = "t-SNE Visualization of Embeddings",
    perplexity: int = 30,
    random_state: int = 42,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    interactive: bool = False
) -> Union[plt.Figure, go.Figure]:
    """
    Create a t-SNE visualization of embeddings colored by class.
    
    Args:
        embeddings: Array of embeddings with shape (n_samples, n_features)
        labels: List of labels for each embedding
        class_names: List of class names
        title: Plot title
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
        interactive: Whether to create an interactive plotly figure
        
    Returns:
        Matplotlib Figure or Plotly Figure object
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels,
        'class': [class_names[label] for label in labels]
    })
    
    if interactive:
        # Create interactive plotly figure
        fig = px.scatter(
            df, x='x', y='y', color='class',
            title=title,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'class': 'Class'},
            color_discrete_map={class_name: CATEGORY_COLORS.get(class_name.lower(), 'gray') 
                              for class_name in class_names},
            width=figsize[0]*100, height=figsize[1]*100
        )
        
        # Customize layout
        fig.update_layout(
            legend=dict(
                title="Class",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save figure if save_path is provided
        if save_path:
            # For interactive plots, save as HTML
            html_path = save_path.replace('.png', '.html')
            fig.write_html(html_path)
            
            # Also save as static image
            fig.write_image(save_path)
        
        return fig
    else:
        # Create static matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each class
        for label, class_name in enumerate(class_names):
            mask = df['label'] == label
            ax.scatter(
                df.loc[mask, 'x'], df.loc[mask, 'y'],
                label=class_name,
                color=CATEGORY_COLORS.get(class_name.lower(), 'gray'),
                alpha=0.7
            )
        
        # Customize plot
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_learning_curve_plot(
    training_history: Dict[str, List[float]],
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a plot of training and validation metrics over epochs.
    
    Args:
        training_history: Dictionary with lists of metrics for each epoch
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss curves
    if 'loss' in training_history and 'val_loss' in training_history:
        epochs = range(1, len(training_history['loss']) + 1)
        ax1.plot(epochs, training_history['loss'], 'bo-', label='Training loss')
        ax1.plot(epochs, training_history['val_loss'], 'ro-', label='Validation loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
    
    # Plot accuracy curves
    if 'accuracy' in training_history and 'val_accuracy' in training_history:
        epochs = range(1, len(training_history['accuracy']) + 1)
        ax2.plot(epochs, training_history['accuracy'], 'bo-', label='Training accuracy')
        ax2.plot(epochs, training_history['val_accuracy'], 'ro-', label='Validation accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_inference_time_plot(
    model_times: Dict[str, float],
    title: str = "Model Inference Time Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a bar plot comparing inference times across models.
    
    Args:
        model_times: Dictionary mapping model names to inference times (ms per sample)
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort models by inference time
    models = sorted(model_times.items(), key=lambda x: x[1])
    model_names = [model[0] for model in models]
    times = [model[1] for model in models]
    
    # Create horizontal bar chart
    bars = ax.barh(model_names, times, color='skyblue')
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(time + max(times)*0.01, i, f'{time:.2f} ms', va='center')
    
    # Customize plot
    ax.set_xlabel('Inference Time (ms per sample)')
    ax.set_ylabel('Model')
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_wordcloud_by_class(
    texts: List[str],
    labels: List[int],
    class_names: List[str],
    title: str = "Word Clouds by Class",
    stopwords: Optional[List[str]] = None,
    max_words: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create word clouds for texts in each class.
    
    Args:
        texts: List of text samples
        labels: List of class labels for each text
        class_names: List of class names
        title: Plot title
        stopwords: List of words to exclude
        max_words: Maximum number of words to include in each word cloud
        save_path: Path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Group texts by class
    texts_by_class = {}
    for text, label in zip(texts, labels):
        if label < len(class_names):
            class_name = class_names[label]
            if class_name not in texts_by_class:
                texts_by_class[class_name] = []
            texts_by_class[class_name].append(text)
    
    # Create figure
    n_classes = len(texts_by_class)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=