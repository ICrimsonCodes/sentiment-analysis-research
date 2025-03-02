"""
evaluation.py - Model evaluation utilities for sentiment analysis

This module provides functions to evaluate and compare transformer-based models
for sentiment analysis tasks, generating detailed performance metrics and visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Set up logging
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SentimentModelEvaluator:
    """Class for evaluating sentiment analysis models."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 128
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the saved model directory
            device: Device to use for inference ('cuda' or 'cpu')
            max_length: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.max_length = max_length
        
        # Set device for evaluation
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            if "num_labels" in self.config:
                self.num_labels = self.config["num_labels"]
            else:
                self.num_labels = 3  # Default: positive, negative, neutral
                
            if "id2label" in self.config:
                self.id2label = self.config["id2label"]
            else:
                self.id2label = {i: f"LABEL_{i}" for i in range(self.num_labels)}
        else:
            logger.warning(f"Config not found at {config_path}. Using default settings.")
            self.config = {}
            self.num_labels = 3
            self.id2label = {i: f"LABEL_{i}" for i in range(self.num_labels)}
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move model to the specified device
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_probabilities: bool = False
    ) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        Predict sentiment labels for a list of texts.
        
        Args:
            texts: List of text samples to classify
            batch_size: Batch size for inference
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of predicted labels or tuple of (labels, probabilities)
        """
        predictions = []
        probabilities = []
        
        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get class predictions
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(batch_predictions.tolist())
                
                # Get probabilities if requested
                if return_probabilities:
                    batch_probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                    probabilities.extend(batch_probs)
        
        if return_probabilities:
            return predictions, np.array(probabilities)
        else:
            return predictions
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        output_dir: Optional[str] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on a test dataset.
        
        Args:
            texts: List of text samples to classify
            labels: List of ground truth labels
            output_dir: Directory to save evaluation results
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Start timing
        start_time = time.time()
        
        # Get predictions and probabilities
        y_pred, y_probas = self.predict(texts, return_probabilities=True)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / len(texts)
        
        # Calculate basic metrics
        accuracy = accuracy_score(labels, y_pred)
        
        if self.num_labels == 2:  # Binary classification
            precision = precision_score(labels, y_pred)
            recall = recall_score(labels, y_pred)
            f1 = f1_score(labels, y_pred)
        else:  # Multi-class classification
            precision = precision_score(labels, y_pred, average='weighted')
            recall = recall_score(labels, y_pred, average='weighted')
            f1 = f1_score(labels, y_pred, average='weighted')
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, y_pred)
        
        # Generate classification report
        report = classification_report(labels, y_pred, target_names=[self.id2label[i] for i in range(self.num_labels)], output_dict=True)
        
        # Store metrics
        metrics = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "inference_time_total": inference_time,
            "inference_time_per_sample": avg_inference_time,
            "num_samples": len(texts)
        }
        
        # Calculate ROC and PR curves for binary classification
        if self.num_labels == 2:
            # ROC curve and AUC
            fpr, tpr, _ = roc_curve(labels, y_probas[:, 1])
            roc_auc = auc(fpr, tpr)
            metrics["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
            
            # Precision-Recall curve and average precision
            precision_curve, recall_curve, _ = precision_recall_curve(labels, y_probas[:, 1])
            avg_precision = average_precision_score(labels, y_probas[:, 1])
            metrics["pr_curve"] = {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
                "average_precision": avg_precision
            }
        
        # Save metrics to file if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metrics as JSON
            with open(os.path.join(output_dir, f"{model_name}_metrics.json"), "w") as f:
                # Filter out non-serializable items
                serializable_metrics = {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "roc", "pr_curve"]}
                json.dump(serializable_metrics, f, indent=2)
            
            # Save confusion matrix as CSV
            np.savetxt(
                os.path.join(output_dir, f"{model_name}_confusion_matrix.csv"),
                cm,
                delimiter=",",
                fmt="%d"
            )
            
            # Create visualizations
            self._create_visualizations(metrics, output_dir, model_name)
        
        logger.info(f"Evaluation results for {model_name}:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Avg. Inference Time: {avg_inference_time*1000:.2f} ms per sample")
        
        return metrics
    
    def _create_visualizations(self, metrics: Dict[str, Any], output_dir: str, model_name: str):
        """
        Create visualizations for evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_dir: Directory to save visualizations
            model_name: Name of the model for file naming
        """
        # Set up plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[self.id2label[i] for i in range(self.num_labels)],
            yticklabels=[self.id2label[i] for i in range(self.num_labels)]
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
        
        # Per-class metrics
        plt.figure(figsize=(12, 6))
        report = metrics["classification_report"]
        class_names = [self.id2label[i] for i in range(self.num_labels)]
        
        class_metrics = {
            "Precision": [report[name]["precision"] for name in class_names],
            "Recall": [report[name]["recall"] for name in class_names],
            "F1-Score": [report[name]["f1-score"] for name in class_names]
        }
        
        df = pd.DataFrame(class_metrics, index=class_names)
        ax = df.plot(kind="bar", figsize=(12, 6))
        ax.set_ylim([0, 1])
        ax.set_xlabel("Sentiment Class")
        ax.set_ylabel("Score")
        ax.set_title(f"Per-class Metrics - {model_name}")
        ax.legend(loc="lower right")
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_class_metrics.png"))
        plt.close()
        
        # ROC curve for binary classification
        if "roc" in metrics:
            plt.figure(figsize=(10, 8))
            plt.plot(
                metrics["roc"]["fpr"],
                metrics["roc"]["tpr"],
                lw=2,
                label=f'ROC curve (AUC = {metrics["roc"]["auc"]:.2f})'
            )
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
            plt.close()
        
        # Precision-Recall curve for binary classification
        if "pr_curve" in metrics:
            plt.figure(figsize=(10, 8))
            plt.plot(
                metrics["pr_curve"]["recall"],
                metrics["pr_curve"]["precision"],
                lw=2,
                label=f'PR curve (AP = {metrics["pr_curve"]["average_precision"]:.2f})'
            )
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_pr_curve.png"))
            plt.close()


def compare_models(
    model_evaluations: Dict[str, Dict[str, Any]],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare the performance of multiple models.
    
    Args:
        model_evaluations: Dictionary mapping model names to their evaluation metrics
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison metrics and best model information
    """
    # Extract key metrics for comparison
    comparison = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1_score": {},
        "inference_time_per_sample": {}
    }
    
    for model_name, metrics in model_evaluations.items():
        for metric in comparison:
            comparison[metric][model_name] = metrics[metric]
    
    # Convert to DataFrames for easier manipulation
    comparison_df = pd.DataFrame(comparison)
    
    # Identify best model for each metric
    best_models = {}
    for metric in comparison:
        if metric == "inference_time_per_sample":
            # Lower is better for inference time
            best_model = comparison_df[metric].idxmin()
            best_value = comparison_df[metric].min()
        else:
            # Higher is better for accuracy, precision, recall, f1
            best_model = comparison_df[metric].idxmax()
            best_value = comparison_df[metric].max()
        
        best_models[metric] = {"model": best_model, "value": best_value}
    
    # Create comparison result
    result = {
        "comparison_table": comparison_df.to_dict(),
        "best_models": best_models
    }
    
    # Save comparison to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison table as CSV
        comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"))
        
        # Save best models as JSON
        with open(os.path.join(output_dir, "best_models.json"), "w") as f:
            json.dump(best_models, f, indent=2)
        
        # Create comparison visualizations
        _create_comparison_visualizations(comparison_df, best_models, output_dir)
    
    return result


def _create_comparison_visualizations(comparison_df: pd.DataFrame, best_models: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Create visualizations comparing model performances.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        best_models: Dictionary with best model information
        output_dir: Directory to save visualizations
    """
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Performance metrics comparison
    performance_metrics = ["accuracy", "precision", "recall", "f1_score"]
    plt.figure(figsize=(14, 8))
    
    # Reshape data for plotting
    plot_df = comparison_df[performance_metrics].reset_index()
    plot_df = pd.melt(plot_df, id_vars=["index"], value_vars=performance_metrics, 
                      var_name="Metric", value_name="Value")
    
    # Create grouped bar chart
    ax = sns.barplot(x="Metric", y="Value", hue="index", data=plot_df)
    
    # Customize plot
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_ylim([0, 1])
    ax.legend(title="Model")
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_performance_comparison.png"))
    plt.close()
    
    # Inference time comparison
    plt.figure(figsize=(10, 6))
    inference_times = comparison_df["inference_time_per_sample"] * 1000  # Convert to ms
    ax = inference_times.plot(kind="bar", figsize=(10, 6))
    ax.set_xlabel("Model")
    ax.set_ylabel("Inference Time (ms per sample)")
    ax.set_title("Model Inference Time Comparison")
    
    # Add value labels on top of bars
    for i, v in enumerate(inference_times):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_inference_time_comparison.png"))
    plt.close()
    
    # Radar chart for model comparison
    plt.figure(figsize=(10, 10))
    
    # Prepare data for radar chart
    metrics = performance_metrics
    num_metrics = len(metrics)
    
    # Angle for each metric
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection (radar chart)
    ax = plt.subplot(111, polar=True)
    
    # Plot each model
    for model in comparison_df.index:
        values = comparison_df.loc[model, metrics].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels for each metric
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Model Comparison Radar Chart")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_radar_comparison.png"))
    plt.close()


def evaluate_model_on_dataset(
    model_path: str,
    texts: List[str],
    labels: List[int],
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model_path: Path to the saved model
        texts: List of text samples to classify
        labels: List of ground truth labels
        model_name: Name of the model for reporting
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    # Initialize evaluator
    evaluator = SentimentModelEvaluator(model_path=model_path)
    
    # Evaluate model
    metrics = evaluator.evaluate(
        texts=texts,
        labels=labels,
        output_dir=output_dir,
        model_name=model_name
    )
    
    return metrics


def evaluate_models_on_dataset(
    model_paths: Dict[str, str],
    texts: List[str],
    labels: List[int],
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple models on the same dataset.
    
    Args:
        model_paths: Dictionary mapping model names to their paths
        texts: List of text samples to classify
        labels: List of ground truth labels
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary mapping model names to their evaluation metrics
    """
    results = {}
    
    for model_name, model_path in model_paths.items():
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = evaluate_model_on_dataset(
            model_path=model_path,
            texts=texts,
            labels=labels,
            model_name=model_name,
            output_dir=output_dir
        )
        
        results[model_name] = metrics
    
    # Compare models if there are multiple
    if len(model_paths) > 1:
        comparison = compare_models(
            model_evaluations=results,
            output_dir=output_dir
        )
        
        # Include comparison in results
        results["comparison"] = comparison
    
    return results


def analyze_errors(
    model_path: str,
    texts: List[str],
    labels: List[int],
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze prediction errors to identify patterns.
    
    Args:
        model_path: Path to the saved model
        texts: List of text samples to classify
        labels: List of ground truth labels
        output_dir: Directory to save analysis results
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with error analysis results
    """
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    # Initialize evaluator
    evaluator = SentimentModelEvaluator(model_path=model_path)
    
    # Get predictions and probabilities
    y_pred, y_probas = evaluator.predict(texts, return_probabilities=True)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "predicted_label": y_pred,
        "correct": np.array(labels) == np.array(y_pred)
    })
    
    # Add prediction confidence
    df["confidence"] = np.max(y_probas, axis=1)
    
    # Add label names
    df["true_label_name"] = df["true_label"].map(evaluator.id2label)
    df["predicted_label_name"] = df["predicted_label"].map(evaluator.id2label)
    
    # Get incorrect predictions
    errors = df[~df["correct"]].copy()
    
    # Analyze error types (confusion between specific classes)
    error_types = {}
    for true_label in range(evaluator.num_labels):
        for pred_label in range(evaluator.num_labels):
            if true_label != pred_label:
                # Count occurrences of this error type
                count = ((df["true_label"] == true_label) & (df["predicted_label"] == pred_label)).sum()
                
                if count > 0:
                    error_key = f"{evaluator.id2label[true_label]}_as_{evaluator.id2label[pred_label]}"
                    error_types[error_key] = {
                        "count": int(count),
                        "percentage": float(count / len(df) * 100),
                        "examples": errors[
                            (errors["true_label"] == true_label) & 
                            (errors["predicted_label"] == pred_label)
                        ]["text"].tolist()[:5]  # Limit to 5 examples
                    }
    
    # Analyze errors by confidence
    confidence_bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    confidence_analysis = {}
    
    for i in range(len(confidence_bins) - 1):
        bin_start = confidence_bins[i]
        bin_end = confidence_bins[i+1]
        bin_key = f"{bin_start:.2f}-{bin_end:.2f}"
        
        # Filter errors in this confidence range
        bin_errors = errors[(errors["confidence"] >= bin_start) & (errors["confidence"] < bin_end)]
        
        if len(bin_errors) > 0:
            confidence_analysis[bin_key] = {
                "count": len(bin_errors),
                "percentage": len(bin_errors) / len(errors) * 100 if len(errors) > 0 else 0,
                "examples": bin_errors["text"].tolist()[:3]  # Limit to 3 examples
            }
    
    # Calculate error rate by text length
    df["text_length"] = df["text"].apply(len)
    length_bins = [0, 50, 100, 150, 200, float('inf')]
    length_analysis = {}
    
    for i in range(len(length_bins) - 1):
        bin_start = length_bins[i]
        bin_end = length_bins[i+1]
        bin_key = f"{bin_start}-{bin_end if bin_end != float('inf') else '∞'}"
        
        # Filter texts in this length range
        bin_texts = df[(df["text_length"] >= bin_start) & (df["text_length"] < bin_end)]
        
        if len(bin_texts) > 0:
            bin_errors = bin_texts[~bin_texts["correct"]]
            error_rate = len(bin_errors) / len(bin_texts) * 100
            
            length_analysis[bin_key] = {
                "total": len(bin_texts),
                "errors": len(bin_errors),
                "error_rate": error_rate
            }
    
    # Collect results
    analysis = {
        "model_name": model_name,
        "total_samples": len(df),
        "total_errors": len(errors),
        "error_rate": len(errors) / len(df) * 100,
        "error_types": error_types,
        "confidence_analysis": confidence_analysis,
        "length_analysis": length_analysis,
        "most_common_errors": errors.groupby(["true_label_name", "predicted_label_name"]).size().reset_index(name="count").sort_values("count", ascending=False).head(5).to_dict(orient="records"),
        "lowest_confidence_correct": df[df["correct"]].nsmallest(5, "confidence")[["text", "true_label_name", "confidence"]].to_dict(orient="records"),
        "highest_confidence_errors": errors.nlargest(5, "confidence")[["text", "true_label_name", "predicted_label_name", "confidence"]].to_dict(orient="records")
    }
    
    # Save analysis to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save analysis as JSON
        with open(os.path.join(output_dir, f"{model_name}_error_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Save error examples as CSV
        errors.to_csv(os.path.join(output_dir, f"{model_name}_errors.csv"), index=False)
        
        # Create error analysis visualizations
        _create_error_analysis_visualizations(df, errors, analysis, output_dir, model_name)
    
    return analysis


def _create_error_analysis_visualizations(
    df: pd.DataFrame,
    errors: pd.DataFrame,
    analysis: Dict[str, Any],
    output_dir: str,
    model_name: str
):
    """
    Create visualizations for error analysis.
    
    Args:
        df: DataFrame with all predictions
        errors: DataFrame with incorrect predictions
        analysis: Dictionary with error analysis results
        output_dir: Directory to save visualizations
        model_name: Name of the model for file naming
    """
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Error types distribution
    plt.figure(figsize=(12, 8))
    error_types = [(k, v["count"]) for k, v in analysis["error_types"].items()]
    error_types.sort(key=lambda x: x[1], reverse=True)
    
    labels = [et[0] for et in error_types]
    counts = [et[1] for et in error_types]
    
    # Plot horizontal bar chart
    plt.barh(labels, counts)
    plt.xlabel("Count")
    plt.ylabel("Error Type")
    plt.title(f"Distribution of Error Types - {model_name}")
    
    # Add count labels
    for i, count in enumerate(counts):
        plt.text(count + 0.5, i, str(count))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_error_types.png"))
    plt.close()
    
    # Confidence distribution for correct vs. incorrect predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="confidence", hue="correct", bins=20, multiple="dodge")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"Confidence Distribution by Prediction Outcome - {model_name}")
    plt.legend(title="Correct", labels=["Incorrect", "Correct"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confidence_distribution.png"))
    plt.close()
    
    # Error rate by text length
    plt.figure(figsize=(10, 6))
    
    # Extract data from length analysis
    length_bins = []
    error_rates = []
    sample_counts = []
    
    for bin_key, bin_data in analysis["length_analysis"].items():
        length_bins.append(bin_key)
        error_rates.append(bin_data["error_rate"])
        sample_counts.append(bin_data["total"])
    
    # Create bar
    # Create bar chart for error rates by text length
    ax = plt.bar(length_bins, error_rates)
    plt.xlabel("Text Length")
    plt.ylabel("Error Rate (%)")
    plt.title(f"Error Rate by Text Length - {model_name}")
    
    # Add labels with sample counts
    for i, (rate, count) in enumerate(zip(error_rates, sample_counts)):
        plt.text(i, rate + 1, f"n={count}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_error_by_length.png"))
    plt.close()
    
    # Confusion matrix focusing on errors
    if len(errors) > 0:
        plt.figure(figsize=(10, 8))
        cm = pd.crosstab(errors["true_label_name"], errors["predicted_label_name"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix of Errors - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_error_confusion_matrix.png"))
        plt.close()
    
    # Error rate by confidence bin
    plt.figure(figsize=(10, 6))
    
    # Extract data from confidence analysis
    conf_bins = []
    conf_counts = []
    
    for bin_key, bin_data in analysis["confidence_analysis"].items():
        conf_bins.append(bin_key)
        conf_counts.append(bin_data["percentage"])
    
    plt.bar(conf_bins, conf_counts)
    plt.xlabel("Confidence Range")
    plt.ylabel("Percentage of Errors (%)")
    plt.title(f"Error Distribution by Confidence - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_error_by_confidence.png"))
    plt.close()


def evaluate_model_on_specific_categories(
    model_path: str,
    texts: List[str],
    labels: List[int],
    categories: List[str],
    category_labels: List[List[int]],
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate model performance on specific text categories.
    
    Args:
        model_path: Path to the saved model
        texts: List of text samples to classify
        labels: List of ground truth labels
        categories: List of category names
        category_labels: List of lists, each containing indices of texts belonging to a category
        output_dir: Directory to save evaluation results
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary mapping categories to evaluation metrics
    """
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    # Initialize evaluator
    evaluator = SentimentModelEvaluator(model_path=model_path)
    
    results = {}
    overall_metrics = None
    
    # Evaluate on all data first
    overall_metrics = evaluator.evaluate(
        texts=texts,
        labels=labels,
        output_dir=output_dir,
        model_name=f"{model_name}_overall"
    )
    
    # Evaluate on each category
    for category, indices in zip(categories, category_labels):
        category_texts = [texts[i] for i in indices]
        category_labels = [labels[i] for i in indices]
        
        if len(category_texts) == 0:
            logger.warning(f"No samples for category '{category}'. Skipping evaluation.")
            continue
        
        category_metrics = evaluator.evaluate(
            texts=category_texts,
            labels=category_labels,
            output_dir=output_dir,
            model_name=f"{model_name}_{category}"
        )
        
        results[category] = category_metrics
    
    # Compare performance across categories
    if output_dir and len(results) > 1:
        # Extract key metrics for each category
        comparison = {
            "accuracy": {},
            "precision": {},
            "recall": {},
            "f1_score": {}
        }
        
        for category, metrics in results.items():
            for metric in comparison:
                comparison[metric][category] = metrics[metric]
        
        # Add overall metrics for comparison
        if overall_metrics:
            for metric in comparison:
                comparison[metric]["overall"] = overall_metrics[metric]
        
        # Convert to DataFrame for visualization
        comparison_df = pd.DataFrame(comparison)
        
        # Create bar charts
        plt.figure(figsize=(14, 8))
        comparison_df.plot(kind="bar", figsize=(14, 8))
        plt.xlabel("Category")
        plt.ylabel("Score")
        plt.title(f"Performance Comparison Across Categories - {model_name}")
        plt.ylim([0, 1])
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_category_comparison.png"))
        plt.close()
        
        # Save comparison as CSV
        comparison_df.to_csv(os.path.join(output_dir, f"{model_name}_category_comparison.csv"))
    
    return results


if __name__ == "__main__":
    # Example usage
    
    # Sample data (replace with your actual data)
    texts = [
        "I love this product, it's amazing!",
        "This movie was terrible and boring.",
        "The service was okay, nothing special.",
        "I'm very disappointed with the quality.",
        "This is the best purchase I've ever made!"
    ]
    
    # Labels: 0 = negative, 1 = neutral, 2 = positive
    labels = [2, 0, 1, 0, 2]
    
    # Example model paths (replace with your actual paths)
    model_paths = {
        "BERT": "./models/bert_model/final_model",
        "RoBERTa": "./models/roberta_model/final_model"
    }
    
    # Evaluate models
    results = evaluate_models_on_dataset(
        model_paths=model_paths,
        texts=texts,
        labels=labels,
        output_dir="./results"
    )
    
    # Print results summary
    for model_name, metrics in results.items():
        if model_name != "comparison":
            print(f"\nResults for {model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # If there's a comparison result, print the best model for each metric
    if "comparison" in results:
        print("\nBest models:")
        for metric, best in results["comparison"]["best_models"].items():
            print(f"  {metric}: {best['model']} ({best['value']:.4f})")