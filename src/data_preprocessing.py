# data_preprocessing.py - Text preprocessing utilities for sentiment analysis

# This module provides functions to preprocess text data for sentiment analysis, 
# with specific handling for social media text features like hashtags, mentions,
# emojis, and other special characters.


import re
import emoji
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    #Text preprocessing class for sentiment analysis tasks.
    
    def __init__(self, 
                 remove_urls=True,
                 remove_mentions=False,
                 process_hashtags=True,
                 process_emojis=True,
                 remove_punctuation=True,
                 convert_lowercase=True,
                 remove_stopwords=False,
                 lemmatize=False,
                 language='english'):
       
# Initialize the text preprocessor with configurable options.
    #Args:
      #remove_urls (bool): Whether to remove URLs from text
      # remove_mentions (bool): Whether to remove @mentions completely
      # process_hashtags (bool): Whether to process #hashtags (remove # and keep text)
      # process_emojis (bool): Whether to replace emojis with their text description
      # remove_punctuation (bool): Whether to remove punctuation
      # convert_lowercase (bool): Whether to convert text to lowercase
      # remove_stopwords (bool): Whether to remove stopwords
      # lemmatize (bool): Whether to perform lemmatization
      # language (str): Language for stopwords (default: 'english')

        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.process_hashtags = process_hashtags
        self.process_emojis = process_emojis
        self.remove_punctuation = remove_punctuation
        self.convert_lowercase = convert_lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.language = language
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        
        # Initialize lemmatizer if needed
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        # Load stopwords if needed
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(self.language))
    
    def process_text(self, text):
        #Process a text string according to the configured options.
        #Args:
          #text (str): The input text to process
          
        if not isinstance(text, str) or text == '':
            return ''
        
        # Handle URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Handle mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub('', text)
        else:
            # Replace @username with @USER to anonymize but keep the mention indicator
            text = self.mention_pattern.sub('@USER', text)
        
        # Handle hashtags
        if self.process_hashtags:
            # Extract the text from hashtags and keep it
            text = self.hashtag_pattern.sub(r'\1', text)
        
        # Handle emojis
        if self.process_emojis:
            text = self.replace_emojis(text)
        
        # Convert to lowercase
        if self.convert_lowercase:
            text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def replace_emojis(self, text):
      
      # Replace emojis with their text description or remove them.
      # Args:
      # text (str): Input text with emojis
      # Returns:
      # str: Text with emojis replaced by their description

        return emoji.demojize(text).replace(':', ' ').replace('_', ' ')
    
    def segment_hashtag(self, hashtag):
      
       # Split hashtag into separate words using simple heuristics.
        
       # Args:
       #     hashtag (str): The hashtag text without the # symbol
            
       # Returns:
       #     str: Space-separated words from the hashtag

        # Method 1: Split by capital letters
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+', hashtag)
        if len(words) > 1:
            return ' '.join(words).lower()
    
        return hashtag


def preprocess_dataset(texts, config=None):
    
   # Preprocess a list of texts with optional custom configuration.
    
   # Args:
   #     texts (list): List of text strings to process
   #     config (dict, optional): Configuration parameters for the TextPreprocessor
        
   # Returns:
   #     list: List of preprocessed text strings

    # Use default configuration if none provided
    if config is None:
        preprocessor = TextPreprocessor()
    else:
        preprocessor = TextPreprocessor(**config)
    
    return [preprocessor.process_text(text) for text in texts]


def load_and_preprocess_data(file_path, text_column, config=None):
  
   # Load data from a CSV file and preprocess the text column.
    
   # Args:
   #     file_path (str): Path to the CSV file
   #     text_column (str): Name of the column containing text to preprocess
   #     config (dict, optional): Configuration parameters for the TextPreprocessor
        
   # Returns:
   #     pandas.DataFrame: DataFrame with preprocessed text

    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset")
        
        preprocessor = TextPreprocessor(**(config or {}))
        df['processed_text'] = df[text_column].apply(preprocessor.process_text)
        
        return df
    
    except ImportError:
        print("Pandas is required to load CSV files. Please install it with 'pip install pandas'")
        return None


if __name__ == "__main__"
      # Example usage
      example_texts = [
          "I absolutely LOVE this product! It's amazing! 😍 #recommended #mustBuy",
          "@user This is disappointing... https://example.com/complaint 😡",
          "The weather is so-so today. Not great, not terrible. #WeatherUpdate",
          "RT @news: Breaking: Important announcement coming! Stay tuned! #BigNews"
      ] 
      
      # Default configuration
      preprocessed_texts = preprocess_dataset(example_texts)
      
      print("Original texts:")
      for text in example_texts:
          print(f"  - {text}")
      
      print("\nPreprocessed texts (default config):")
      for text in preprocessed_texts:
          print(f"  - {text}")
      
      # Custom configuration
      custom_config = {
          'remove_urls': True,
          'remove_mentions': True,
          'process_hashtags': True,
          'process_emojis': True,
          'remove_punctuation': True,
          'convert_lowercase': True,
          'remove_stopwords': True,
          'lemmatize': True
      }
      
      preprocessed_texts_custom = preprocess_dataset(example_texts, custom_config)
      
      print("\nPreprocessed texts (custom config with stopword removal and lemmatization):")
      for text in preprocessed_texts_custom:  
          print(f"  - {text}")