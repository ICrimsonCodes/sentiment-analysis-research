"""
Data preprocessing script for sentiment analysis research.
This script processes the Stanford Sentiment Treebank and Reddit datasets.
"""

import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
SST_FILE = RAW_DATA_DIR / "stanford_sentiment_treebank.csv"
REDDIT_FILE = RAW_DATA_DIR / "reddit_data.csv"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)


def download_nltk_resources():
    """Download required NLTK resources if not present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def clean_text(text):
    """
    Clean text by removing special characters, links, and extra whitespace.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers, keeping only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


def process_sst_dataset():
    """
    Process the Stanford Sentiment Treebank dataset.
    
    Returns:
        tuple: (train_data, val_data, test_data) as pandas DataFrames
    """
    logger.info("Processing Stanford Sentiment Treebank dataset...")
    
    try:
        # Load SST dataset
        df = pd.read_csv(SST_FILE)
        logger.info(f"Loaded {len(df)} rows from SST dataset")
        
        # Basic info about the dataset
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        logger.info(f"Label distribution: {df['labels'].value_counts().to_dict()}")
        
        # Clean text
        logger.info("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Remove stopwords
        logger.info("Removing stopwords...")
        df['processed_text'] = df['cleaned_text'].apply(remove_stopwords)
        
        # Drop rows with empty text after processing
        df = df.dropna(subset=['processed_text'])
        df = df[df['processed_text'].str.strip() != '']
        logger.info(f"{len(df)} rows after cleaning")
        
        # Split the data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['labels'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['labels'])
        
        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Save processed data
        train_df.to_csv(PROCESSED_DATA_DIR / "sst_train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / "sst_val.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "sst_test.csv", index=False)
        
        return train_df, val_df, test_df
    
    except Exception as e:
        logger.error(f"Error processing SST dataset: {e}")
        raise


def process_reddit_data():
    """
    Process the Reddit dataset in CSV format.
    
    Returns:
        tuple: (train_data, val_data, test_data) as pandas DataFrames
    """
    logger.info("Processing Reddit dataset...")
    
    try:
        # Check if reddit data file exists
        if not REDDIT_FILE.exists():
            logger.error(f"Reddit data file not found: {REDDIT_FILE}")
            return None, None, None
        
        # Load Reddit dataset
        df = pd.read_csv(REDDIT_FILE)
        logger.info(f"Loaded {len(df)} rows from Reddit dataset")
        logger.info(f"Loaded {len(df)} rows from Reddit dataset")
        
        # Clean text
        logger.info("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Remove stopwords
        logger.info("Removing stopwords...")
        df['processed_text'] = df['cleaned_text'].apply(remove_stopwords)
        
        # Drop rows with empty text after processing
        df = df.dropna(subset=['processed_text'])
        df = df[df['processed_text'].str.strip() != '']
        logger.info(f"{len(df)} rows after cleaning")
        
        # Split the data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['labels'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['labels'])
        
        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Save processed data
        train_df.to_csv(PROCESSED_DATA_DIR / "reddit_train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / "reddit_val.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "reddit_test.csv", index=False)
        
        return train_df, val_df, test_df
    
    except Exception as e:
        logger.error(f"Error processing Reddit dataset: {e}")
        raise


def combine_datasets(sst_data, reddit_data):
    """
    Combine SST and Reddit datasets.
    
    Args:
        sst_data (tuple): (train, val, test) DataFrames for SST
        reddit_data (tuple): (train, val, test) DataFrames for Reddit
        
    Returns:
        tuple: Combined (train, val, test) DataFrames
    """
    logger.info("Combining datasets...")
    
    combined_data = []
    
    for i, (sst_df, reddit_df) in enumerate(zip(sst_data, reddit_data)):
        if sst_df is not None and reddit_df is not None:
            # Add source column to identify the origin
            sst_df = sst_df.copy()
            reddit_df = reddit_df.copy()
            
            sst_df['source'] = 'sst'
            reddit_df['source'] = 'reddit'
            
            # Combine
            combined_df = pd.concat([sst_df, reddit_df], ignore_index=True)
            
            # Save
            split_name = ['train', 'val', 'test'][i]
            combined_df.to_csv(PROCESSED_DATA_DIR / f"combined_{split_name}.csv", index=False)
            
            combined_data.append(combined_df)
        else:
            combined_data.append(None)
    
    return tuple(combined_data)


def save_dataset_stats(train_df, val_df, test_df, dataset_name):
    """
    Save statistics about the dataset splits.
    
    Args:
        train_df, val_df, test_df: DataFrames for each split
        dataset_name: Name of the dataset
    """
    stats = {
        'dataset': dataset_name,
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'label_distribution': {
            'train': train_df['labels'].value_counts().to_dict(),
            'val': val_df['labels'].value_counts().to_dict(),
            'test': test_df['labels'].value_counts().to_dict()
        },
        'avg_text_length': {
            'train': train_df['processed_text'].str.len().mean(),
            'val': val_df['processed_text'].str.len().mean(),
            'test': test_df['processed_text'].str.len().mean()
        }
    }
    
    with open(PROCESSED_DATA_DIR / f"{dataset_name}_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved stats for {dataset_name} dataset")


def main():
    """Main function to run data preprocessing."""
    logger.info("Starting data preprocessing")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Process SST dataset
    sst_data = process_sst_dataset()
    if all(sst_data):
        save_dataset_stats(*sst_data, 'sst')
    
    # Process Reddit dataset
    reddit_data = process_reddit_data()
    if all(reddit_data):
        save_dataset_stats(*reddit_data, 'reddit')
    
    # Combine datasets if both are available
    if all(sst_data) and all(reddit_data):
        combined_data = combine_datasets(sst_data, reddit_data)
        if all(combined_data):
            save_dataset_stats(*combined_data, 'combined')
    
    logger.info("Data preprocessing completed")


if __name__ == "__main__":
    main()