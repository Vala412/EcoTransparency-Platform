"""
Data ingestion and preprocessing module
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import cv2
from PIL import Image
import spacy
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Main data ingestion and preprocessing pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.nlp = None
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def ingest_product_data(self, data_path: str) -> pd.DataFrame:
        """
        Ingest product lifecycle data from various sources
        
        Args:
            data_path: Path to the data file (CSV, JSON, or Excel)
            
        Returns:
            Processed DataFrame with product data
        """
        try:
            # Determine file type and read accordingly
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            elif data_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            # Standardize column names
            df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
            
            # Basic data cleaning
            df = self._clean_product_data(df)
            
            logger.info(f"Ingested {len(df)} products from {data_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting data from {data_path}: {str(e)}")
            raise
    
    def _clean_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize product data"""
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        text_columns = df.select_dtypes(include=['object']).columns
        df[text_columns] = df[text_columns].fillna('Unknown')
        
        # Standardize text data
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        return df
    
    def preprocess_text_data(self, texts: List[str]) -> Dict[str, Any]:
        """
        Preprocess text data for NLP analysis
        
        Args:
            texts: List of text strings to process
            
        Returns:
            Dictionary with processed text features
        """
        if not self.nlp:
            logger.warning("NLP model not available. Skipping text preprocessing.")
            return {"texts": texts, "features": None}
        
        processed_texts = []
        entities = []
        sentiments = []
        
        for text in texts:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract cleaned text
            cleaned_text = " ".join([token.lemma_.lower() for token in doc 
                                   if not token.is_stop and not token.is_punct and token.is_alpha])
            processed_texts.append(cleaned_text)
            
            # Extract entities
            text_entities = [(ent.text, ent.label_) for ent in doc.ents]
            entities.append(text_entities)
            
            # Simple sentiment analysis (placeholder)
            sentiment_score = self._calculate_sentiment(text)
            sentiments.append(sentiment_score)
        
        # Create TF-IDF features
        if 'text_tfidf' not in self.vectorizers:
            self.vectorizers['text_tfidf'] = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_features = self.vectorizers['text_tfidf'].fit_transform(processed_texts)
        else:
            tfidf_features = self.vectorizers['text_tfidf'].transform(processed_texts)
        
        return {
            "original_texts": texts,
            "processed_texts": processed_texts,
            "entities": entities,
            "sentiments": sentiments,
            "tfidf_features": tfidf_features,
            "feature_names": self.vectorizers['text_tfidf'].get_feature_names_out()
        }
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score (placeholder implementation)"""
        # Simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'eco-friendly', 'sustainable']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'wasteful', 'polluting']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words
    
    def preprocess_images(self, image_paths: List[str], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess product images for computer vision
        
        Args:
            image_paths: List of image file paths
            target_size: Target size for image resizing
            
        Returns:
            Preprocessed image array
        """
        images = []
        
        for image_path in image_paths:
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize image
                image = cv2.resize(image, target_size)
                
                # Normalize pixel values
                image = image.astype(np.float32) / 255.0
                
                images.append(image)
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                continue
        
        return np.array(images)
    
    def format_time_series_data(self, df: pd.DataFrame, date_column: str, value_column: str) -> pd.DataFrame:
        """
        Format data for time series analysis
        
        Args:
            df: DataFrame with time series data
            date_column: Name of the date column
            value_column: Name of the value column
            
        Returns:
            Formatted time series DataFrame
        """
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Set date as index
        df.set_index(date_column, inplace=True)
        
        # Handle missing values in time series
        df[value_column] = df[value_column].interpolate(method='linear')
        
        # Add time-based features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        
        # Add lag features
        for lag in [1, 7, 30]:
            df[f'{value_column}_lag_{lag}'] = df[value_column].shift(lag)
        
        # Add rolling statistics
        for window in [7, 30]:
            df[f'{value_column}_rolling_mean_{window}'] = df[value_column].rolling(window=window).mean()
            df[f'{value_column}_rolling_std_{window}'] = df[value_column].rolling(window=window).std()
        
        return df
    
    def create_user_profiles(self, interaction_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create user behavioral profiles from interaction data
        
        Args:
            interaction_data: DataFrame with user interaction history
            
        Returns:
            DataFrame with user profiles
        """
        # Group by user
        user_profiles = interaction_data.groupby('user_id').agg({
            'product_id': 'count',  # Total interactions
            'rating': ['mean', 'std'],  # Rating statistics
            'interaction_type': lambda x: x.value_counts().to_dict(),  # Interaction type distribution
            'category': lambda x: x.value_counts().to_dict(),  # Category preferences
            'sustainability_score': ['mean', 'std'],  # Sustainability preferences
            'price': ['mean', 'std']  # Price preferences
        }).reset_index()
        
        # Flatten column names
        user_profiles.columns = ['_'.join(col).strip() if col[1] else col[0] for col in user_profiles.columns]
        user_profiles.rename(columns={'user_id_': 'user_id'}, inplace=True)
        
        # Calculate additional metrics
        user_profiles['sustainability_preference'] = user_profiles['sustainability_score_mean'] / user_profiles['sustainability_score_mean'].max()
        user_profiles['price_sensitivity'] = 1 - (user_profiles['price_mean'] / user_profiles['price_mean'].max())
        
        return user_profiles
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str], fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df: DataFrame with features to normalize
            feature_columns: List of column names to normalize
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with normalized features
        """
        df_normalized = df.copy()
        
        for col in feature_columns:
            if col in df.columns:
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                    df_normalized[col] = self.scalers[col].fit_transform(df[col].values.reshape(-1, 1)).flatten()
                else:
                    if col in self.scalers:
                        df_normalized[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()
        
        return df_normalized
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str], fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder
        
        Args:
            df: DataFrame with categorical features
            categorical_columns: List of column names to encode
            fit: Whether to fit the encoder (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.encoders:
                        try:
                            df_encoded[col] = self.encoders[col].transform(df[col].astype(str))
                        except ValueError:
                            # Handle unseen categories
                            df_encoded[col] = 0
        
        return df_encoded

class ExternalDataConnector:
    """Connector for external data sources"""
    
    def __init__(self):
        self.carbon_intensity_data = None
        self.supply_chain_data = None
    
    def fetch_carbon_intensity_data(self, region: str = "global") -> Dict[str, float]:
        """
        Fetch carbon intensity data for different energy sources
        
        Args:
            region: Geographic region for data
            
        Returns:
            Dictionary with carbon intensity values
        """
        # Placeholder data (in real implementation, this would connect to external APIs)
        carbon_intensity = {
            "coal": 820,  # kg CO2e/MWh
            "natural_gas": 490,
            "oil": 778,
            "nuclear": 12,
            "hydro": 24,
            "wind": 11,
            "solar": 41,
            "biomass": 230,
            "geothermal": 38
        }
        
        return carbon_intensity
    
    def fetch_supply_chain_data(self, product_id: str) -> Dict[str, Any]:
        """
        Fetch supply chain metadata for a product
        
        Args:
            product_id: Product identifier
            
        Returns:
            Dictionary with supply chain information
        """
        # Placeholder implementation
        return {
            "suppliers": [],
            "transportation_routes": [],
            "certifications": [],
            "labor_practices": {},
            "environmental_compliance": {}
        }
    
    def fetch_market_data(self, product_category: str) -> Dict[str, Any]:
        """
        Fetch market data for demand forecasting
        
        Args:
            product_category: Product category
            
        Returns:
            Dictionary with market data
        """
        # Placeholder implementation
        return {
            "historical_demand": [],
            "seasonal_patterns": {},
            "market_trends": {},
            "economic_indicators": {}
        }
