"""
Model training script for sentiment analysis.
This script trains transformer-based models on the preprocessed datasets.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

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

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


class ModelTrainer:
    """Class for training sentiment analysis models."""
    
    def __init__(
        self,
        model_type,
        model_name,
        dataset_name,
        pretrained_model=None,
        num_labels=3,
        device=None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_type (str): Type of model architecture (bert, roberta, etc.)
            model_name (str): Specific version or variant name
            dataset_name (str): Name of dataset to train on (sst, reddit, combined)
            pretrained_model (str): HuggingFace model ID to use as base
            num_labels (int): Number of sentiment classes
            device (str): Device to train on (cuda, cpu)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_labels = num_labels
        
        # Set default pretrained model if not provided
        if pretrained_model is None:
            if model_type == 'bert':
                self.pretrained_model = 'bert-base-uncased'
            elif model_type == 'roberta':
                self.pretrained_model = 'roberta-base'
            else:
                self.pretrained_model = model_type
        else:
            self.pretrained_model = pretrained_model
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Training {model_type}/{model_name} on {dataset_name} dataset using {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model,
            num_labels=num_labels
        ).to(self.device)
        
        # Create the model output directory
        self.output_dir = MODELS_DIR / model_type / model_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Training metrics tracking
        self.train_stats = {
            'model_type': model_type,
            'model_name': model_name,
            'dataset': dataset_name,
            'pretrained_model': self.pretrained_model,
            'num_labels': num_labels,
            'training_started': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'epochs': [],
            'training_time': None
        }
    
    def prepare_data(self, batch_size=16, max_length=128):
        """
        Prepare training and validation data for model training.
        
        Args:
            batch_size (int): Batch size for training
            max_length (int): Maximum sequence length for tokenization
        
        Returns:
            tuple: Training and validation DataLoaders
        """
        # Load train and validation data
        train_path = PROCESSED_DATA_DIR / f"{self.dataset_name}_train.csv"
        val_path = PROCESSED_DATA_DIR / f"{self.dataset_name}_val.csv"
        
        if not train_path.exists() or not val_path.exists():
            logger.error(f"Dataset files not found: {train_path} or {val_path}")
            raise FileNotFoundError(f"Dataset files not found")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        logger.info(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples")
        
        # Extract texts and labels
        train_texts = train_df['processed_text'].tolist()
        val_texts = val_df['processed_text'].tolist()
        
        train_labels = train_df['labels'].tolist()
        val_labels = val_df['labels'].tolist()
        
        # Convert labels to numeric if they are not already
        if not isinstance(train_labels[0], (int, float, np.number)):
            # Create a label mapping
            unique_labels = sorted(set(train_labels + val_labels))
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            
            train_labels = [self.label_map[label] for label in train_labels]
            val_labels = [self.label_map[label] for label in val_labels]
            
            # Save the label mapping
            with open(self.output_dir / "label_map.json", "w") as f:
                json.dump(self.label_map, f, indent=2)
            
            logger.info(f"Created label mapping: {self.label_map}")
        
        # Tokenize texts
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Create TensorDatasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_labels)
        )
        
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(val_labels)
        )
        
        # Create DataLoaders
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        return train_dataloader, val_dataloader
    
    def train(self, 
              epochs=4, 
              batch_size=16, 
              learning_rate=2e-5,
              max_length=128,
              weight_decay=0.01,
              warmup_steps=0,
              save_every_epoch=True):
        """
        Train the model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            max_length (int): Maximum sequence length
            weight_decay (float): Weight decay for regularization
            warmup_steps (int): Number of warmup steps for learning rate scheduler
            save_every_epoch (bool): Whether to save the model after each epoch
            
        Returns:
            dict: Training statistics
        """
        # Prepare data
        train_dataloader, val_dataloader = self.prepare_data(
            batch_size=batch_size,
            max_length=max_length
        )
        
        # Save training config
        self.train_stats['config'] = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_length': max_length,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'train_samples': len(train_dataloader.dataset),
            'val_samples': len(val_dataloader.dataset)
        }
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_dataloader) * epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        start_time = time.time()
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                # Zero gradients
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Log progress
                if (step + 1) % 50 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} | Step {step + 1}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, labels = batch
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_dataloader)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
            logger.info(f"Validation F1: {val_f1:.4f}")
            
            # Save epoch stats
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1
            }
            
            self.train_stats['epochs'].append(epoch_stats)
            
            # Save model if it's the best so far
            if val_f1 > best_val_f1:
                logger.info(f"New best model with F1: {val_f1:.4f} (previous: {best_val_f1:.4f})")
                best_val_f1 = val_f1
                
                # Save the best model
                self.model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
                
                # Save best model info
                self.train_stats['best_model'] = {
                    'epoch': epoch + 1,
                    'val_f1': val_f1,
                    'val_accuracy': val_accuracy
                }
            
            # Optionally save after each epoch
            if save_every_epoch:
                epoch_dir = self.output_dir / f"epoch_{epoch + 1}"
                epoch_dir.mkdir(exist_ok=True)
                
                self.model.save_pretrained(epoch_dir)
                
                # We don't need to save tokenizer for every epoch
                # Just save a reference to the main tokenizer
                with open(epoch_dir / "tokenizer_reference.txt", "w") as f:
                    f.write(f"Tokenizer is saved in the parent directory: {self.output_dir}")
        
        # Calculate total training time
        training_time = time.time() - start_time
        self.train_stats['training_time'] = training_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final training stats
        self.train_stats['training_completed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(self.train_stats, f, indent=2)
        
        return self.train_stats


def train_models(config):
    """
    Train multiple models based on configuration.
    
    Args:
        config (dict): Configuration for model training
        
    Returns:
        list: Training statistics for all trained models
    """
    stats = []
    
    for model_config in config['models']:
        # Extract model configuration
        model_type = model_config['type']
        model_name = model_config['name']
        dataset_name = model_config.get('dataset', 'combined')
        pretrained_model = model_config.get('pretrained_model', None)
        num_labels = model_config.get('num_labels', 3)
        
        # Create directory for this model type if it doesn't exist
        (MODELS_DIR / model_type).mkdir(exist_ok=True, parents=True)
        
        # Initialize trainer
        trainer = ModelTrainer(
            model_type=model_type,
            model_name=model_name,
            dataset_name=dataset_name,
            pretrained_model=pretrained_model,
            num_labels=num_labels
        )
        
        # Extract training parameters
        train_params = {
            'epochs': model_config.get('epochs', config.get('epochs', 4)),
            'batch_size': model_config.get('batch_size', config.get('batch_size', 16)),
            'learning_rate': model_config.get('learning_rate', config.get('learning_rate', 2e-5)),
            'max_length': model_config.get('max_length', config.get('max_length', 128)),
            'weight_decay': model_config.get('weight_decay', config.get('weight_decay', 0.01)),
            'warmup_steps': model_config.get('warmup_steps', config.get('warmup_steps', 0)),
            'save_every_epoch': model_config.get('save_every_epoch', config.get('save_every_epoch', True))
        }
        
        # Train the model
        try:
            logger.info(f"Training {model_type}/{model_name} on {dataset_name} dataset")
            model_stats = trainer.train(**train_params)
            stats.append(model_stats)
            logger.info(f"Completed training {model_type}/{model_name}")
        except Exception as e:
            logger.error(f"Error training {model_type}/{model_name}: {e}")
    
    return stats


def main():
    """Main function to train models."""
    logger.info("Starting model training")
    
    # Define training configuration
    config = {
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'max_length': 128,
        'save_every_epoch': True,
        'models': [
            {
                'type': 'bert',
                'name': 'bert-base-sst',
                'dataset': 'sst',
                'pretrained_model': 'bert-base-uncased',
                'epochs': 4
            },
            {
                'type': 'bert',
                'name': 'bert-base-reddit',
                'dataset': 'reddit',
                'pretrained_model': 'bert-base-uncased',
                'epochs': 4
            },
            {
                'type': 'bert',
                'name': 'bert-base-combined',
                'dataset': 'combined',
                'pretrained_model': 'bert-base-uncased',
                'epochs': 4
            },
            {
                'type': 'roberta',
                'name': 'roberta-base-sst',
                'dataset': 'sst',
                'pretrained_model': 'roberta-base',
                'epochs': 3
            },
            {
                'type': 'roberta',
                'name': 'roberta-base-reddit',
                'dataset': 'reddit',
                'pretrained_model': 'roberta-base',
                'epochs': 3
            },
            {
                'type': 'roberta',
                'name': 'roberta-base-combined',
                'dataset': 'combined',
                'pretrained_model': 'roberta-base',
                'epochs': 3
            }
        ]
    }
    
    # Save training configuration
    with open(MODELS_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Train models
    stats = train_models(config)
    
    # Save summary of all training runs
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Model training completed")


if __name__ == "__main__":
    main()