"""
Evaluation script for sentiment analysis models.
This script evaluates trained models on test datasets and generates performance reports.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report,
    roc_curve, 
    auc
)
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
import pickle
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


class ModelEvaluator:
    """Class for evaluating sentiment analysis models."""
    
    def __init__(self, model_type, model_name, num_labels=3):
        """
        Initialize the evaluator.
        
        Args:
            model_type (str): Type of model (bert, roberta, etc.)
            model_name (str): Name/version of the model
            num_labels (int): Number of sentiment classes
        """
        self.model_type = model_type
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model_path = MODELS_DIR / model_type / model_name
        
        if not self.model_path.exists():
            raise ValueError(f"Model not found at {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=num_labels
        ).to(self.device)
        
        logger.info(f"Loaded {model_type}/{model_name} model on {self.device}")
    
    def prepare_data(self, data_path, batch_size=16, max_length=128):
        """
        Prepare the dataset for evaluation.
        
        Args:
            data_path (Path): Path to the CSV data file
            batch_size (int): Batch size for evaluation
            max_length (int): Maximum sequence length
            
        Returns:
            DataLoader: DataLoader for the dataset
        """
        df = pd.read_csv(data_path)
        texts = df['processed_text'].tolist()
        labels = df['labels'].tolist()
        
        # Check if labels are numeric
        if not isinstance(labels[0], (int, float, np.number)):
            # Create a label mapping if needed
            unique_labels = sorted(set(labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            labels = [label_map[label] for label in labels]
            self.label_map = label_map
            self.inv_label_map = {v: k for k, v in label_map.items()}
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Convert to TensorDataset
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return dataloader, df
    
    def evaluate(self, dataloader):
        """
        Evaluate the model on the given dataloader.
        
        Args:
            dataloader (DataLoader): DataLoader with test data
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []  # For ROC curves
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision_macro': precision_score(all_labels, all_preds, average='macro'),
            'recall_macro': recall_score(all_labels, all_preds, average='macro'),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        }
        
        if self.num_labels == 2:  # Binary classification
            metrics['precision'] = precision_score(all_labels, all_preds)
            metrics['recall'] = recall_score(all_labels, all_preds)
            metrics['f1'] = f1_score(all_labels, all_preds)
        else:  # Multi-class
            # Add per-class metrics
            class_report = classification_report(all_labels, all_preds, output_dict=True)
            metrics['per_class'] = {}
            
            for i in range(self.num_labels):
                class_name = str(i)
                if hasattr(self, 'inv_label_map'):
                    class_name = self.inv_label_map[i]
                
                metrics['per_class'][class_name] = {
                    'precision': class_report[str(i)]['precision'],
                    'recall': class_report[str(i)]['recall'],
                    'f1': class_report[str(i)]['f1-score'],
                    'support': int(class_report[str(i)]['support'])
                }
        
        # Add raw predictions and labels for further analysis
        metrics['predictions'] = all_preds.tolist()
        metrics['labels'] = all_labels.tolist()
        metrics['probabilities'] = all_probs.tolist()
        
        return metrics
    
    def save_confusion_matrix(self, labels, predictions, save_path):
        """
        Generate and save a confusion matrix visualization.
        
        Args:
            labels (array): True labels
            predictions (array): Predicted labels
            save_path (Path): Path to save the confusion matrix image
        """
        plt.figure(figsize=(10, 8))
        
        # Create label names for the confusion matrix
        label_names = [str(i) for i in range(self.num_labels)]
        if hasattr(self, 'inv_label_map'):
            label_names = [self.inv_label_map[i] for i in range(self.num_labels)]
        
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.model_type}/{self.model_name}')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def save_roc_curve(self, labels, probabilities, save_path):
        """
        Generate and save ROC curves for multi-class classification.
        
        Args:
            labels (array): True labels
            probabilities (array): Predicted probabilities
            save_path (Path): Path to save the ROC curve image
        """
        plt.figure(figsize=(10, 8))
        
        # Create one-hot encoded labels for multi-class ROC
        y_true = np.zeros((len(labels), self.num_labels))
        for i, label in enumerate(labels):
            y_true[i, label] = 1
        
        # Plot ROC curve for each class
        for i in range(self.num_labels):
            fpr, tpr, _ = roc_curve(y_true[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            
            label_name = str(i)
            if hasattr(self, 'inv_label_map'):
                label_name = self.inv_label_map[i]
                
            plt.plot(
                fpr, 
                tpr, 
                label=f'{label_name} (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {self.model_type}/{self.model_name}')
        plt.legend(loc='lower right')
        plt.savefig(save_path)
        plt.close()
    
    def evaluate_dataset(self, dataset_name):
        """
        Evaluate the model on a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset (sst, reddit, combined)
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating {self.model_type}/{self.model_name} on {dataset_name} dataset")
        
        test_file = PROCESSED_DATA_DIR / f"{dataset_name}_test.csv"
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            return None
        
        # Prepare data
        dataloader, df = self.prepare_data(test_file)
        
        # Run evaluation
        metrics = self.evaluate(dataloader)
        
        # Add metadata
        metrics['model_type'] = self.model_type
        metrics['model_name'] = self.model_name
        metrics['dataset'] = dataset_name
        metrics['num_samples'] = len(df)
        metrics['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save metrics
        results_dir = RESULTS_DIR / self.model_type / self.model_name
        results_dir.mkdir(exist_ok=True, parents=True)
        
        metrics_file = results_dir / f"{dataset_name}_metrics.json"
        
        # Remove predictions, labels, and probabilities from the JSON output
        # (but keep them in the metrics dict for visualization)
        metrics_json = {k: v for k, v in metrics.items() 
                       if k not in ['predictions', 'labels', 'probabilities']}
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        # Generate and save visualizations
        self.save_confusion_matrix(
            metrics['labels'],
            metrics['predictions'],
            results_dir / f"{dataset_name}_confusion_matrix.png"
        )
        
        self.save_roc_curve(
            metrics['labels'],
            metrics['probabilities'],
            results_dir / f"{dataset_name}_roc_curve.png"
        )
        
        logger.info(f"Evaluation completed and results saved to {results_dir}")
        
        return metrics
    
    def generate_report(self, metrics, output_path):
        """
        Generate a detailed evaluation report.
        
        Args:
            metrics (dict): Evaluation metrics
            output_path (Path): Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write(f"# Evaluation Report: {self.model_type}/{self.model_name}\n\n")
            f.write(f"**Dataset**: {metrics['dataset']}\n")
            f.write(f"**Samples**: {metrics['num_samples']}\n")
            f.write(f"**Date**: {metrics['timestamp']}\n\n")
            
            f.write("## Overall Metrics\n\n")
            f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"- Macro Precision: {metrics['precision_macro']:.4f}\n")
            f.write(f"- Macro Recall: {metrics['recall_macro']:.4f}\n")
            f.write(f"- Macro F1: {metrics['f1_macro']:.4f}\n\n")
            
            if 'per_class' in metrics:
                f.write("## Per-Class Metrics\n\n")
                f.write("| Class | Precision | Recall | F1 | Support |\n")
                f.write("|-------|-----------|--------|----|---------|\n")
                
                for class_name, class_metrics in metrics['per_class'].items():
                    f.write(f"| {class_name} | {class_metrics['precision']:.4f} | ")
                    f.write(f"{class_metrics['recall']:.4f} | {class_metrics['f1']:.4f} | ")
                    f.write(f"{class_metrics['support']} |\n")
                
                f.write("\n")
            
            # Add paths to visualizations
            f.write("## Visualizations\n\n")
            f.write(f"- [Confusion Matrix]({metrics['dataset']}_confusion_matrix.png)\n")
            f.write(f"- [ROC Curve]({metrics['dataset']}_roc_curve.png)\n\n")
            
            f.write("## Notes\n\n")
            f.write("- Add any additional observations or notes here\n")
    
    def run_evaluation_pipeline(self, datasets=None):
        """
        Run the full evaluation pipeline on multiple datasets.
        
        Args:
            datasets (list): List of dataset names to evaluate
        
        Returns:
            dict: Dictionary of evaluation results by dataset
        """
        if datasets is None:
            datasets = ['sst', 'reddit', 'combined']
        
        results = {}
        
        for dataset_name in datasets:
            metrics = self.evaluate_dataset(dataset_name)
            
            if metrics:
                results[dataset_name] = metrics
                
                # Generate report
                report_path = RESULTS_DIR / self.model_type / self.model_name / f"{dataset_name}_report.md"
                self.generate_report(metrics, report_path)
        
        return results


def compare_models(model_evaluations, output_path):
    """
    Compare multiple models and generate a comparison report.
    
    Args:
        model_evaluations (dict): Dictionary mapping model names to evaluation results
        output_path (Path): Path to save the comparison report
    """
    if not model_evaluations:
        logger.error("No model evaluations to compare")
        return
    
    # Group by dataset
    datasets = set()
    for model_results in model_evaluations.values():
        datasets.update(model_results.keys())
    
    with open(output_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for dataset in sorted(datasets):
            f.write(f"## Dataset: {dataset}\n\n")
            
            # Create comparison table for main metrics
            f.write("### Overall Metrics\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 |\n")
            f.write("|-------|----------|-----------|--------|----|\n")
            
            for model, results in model_evaluations.items():
                if dataset in results:
                    metrics = results[dataset]
                    f.write(f"| {model} | {metrics['accuracy']:.4f} | ")
                    f.write(f"{metrics['precision_macro']:.4f} | ")
                    f.write(f"{metrics['recall_macro']:.4f} | ")
                    f.write(f"{metrics['f1_macro']:.4f} |\n")
            
            f.write("\n")


def list_available_models():
    """
    List all available models in the models directory.
    
    Returns:
        dict: Dictionary mapping model types to available model names
    """
    models = {}
    
    for model_type_dir in MODELS_DIR.iterdir():
        if model_type_dir.is_dir():
            model_type = model_type_dir.name
            models[model_type] = []
            
            for model_dir in model_type_dir.iterdir():
                if model_dir.is_dir():
                    models[model_type].append(model_dir.name)
    
    return models


def main():
    """Main function to run model evaluation."""
    logger.info("Starting model evaluation")
    
    # List available models
    available_models = list_available_models()
    logger.info(f"Available models: {available_models}")
    
    if not available_models:
        logger.error("No models found for evaluation")
        return
    
    # Track all evaluations for comparison
    all_evaluations = {}
    
    # Evaluate each model
    for model_type, model_names in available_models.items():
        for model_name in model_names:
            model_key = f"{model_type}/{model_name}"
            
            try:
                # Determine number of labels from model config or default to 3
                num_labels = 3  # Default for SST (negative, neutral, positive)
                
                # Initialize evaluator
                evaluator = ModelEvaluator(model_type, model_name, num_labels)
                
                # Run evaluation pipeline
                results = evaluator.run_evaluation_pipeline()
                
                all_evaluations[model_key] = results
            
            except Exception as e:
                logger.error(f"Error evaluating {model_key}: {e}")
    
    # Generate comparison report
    if all_evaluations:
        compare_models(
            all_evaluations,
            RESULTS_DIR / "model_comparison.md"
        )
    
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()