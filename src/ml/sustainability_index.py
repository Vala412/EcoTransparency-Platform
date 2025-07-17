"""
Sustainability Index Generation Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SustainabilityIndexGenerator:
    """Generate sustainability scores using ML models"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.feature_importance = {}
        self.scaler = None
        self.model_type = "random_forest"
        self.version = "1.0.0"
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for sustainability scoring
        
        Args:
            df: DataFrame with product data
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Carbon footprint features
        features_df['carbon_per_kg'] = features_df.get('carbon_footprint', 0) / (features_df.get('weight', 1) + 1e-8)
        features_df['carbon_per_dollar'] = features_df.get('carbon_footprint', 0) / (features_df.get('price', 1) + 1e-8)
        
        # Energy efficiency features
        features_df['energy_per_kg'] = features_df.get('energy_consumption', 0) / (features_df.get('weight', 1) + 1e-8)
        features_df['energy_per_dollar'] = features_df.get('energy_consumption', 0) / (features_df.get('price', 1) + 1e-8)
        
        # Water usage features
        features_df['water_per_kg'] = features_df.get('water_usage', 0) / (features_df.get('weight', 1) + 1e-8)
        features_df['water_per_dollar'] = features_df.get('water_usage', 0) / (features_df.get('price', 1) + 1e-8)
        
        # Transportation features
        features_df['transport_emissions'] = (
            features_df.get('transportation_distance', 0) * 
            features_df.get('transport_emission_factor', 0.1)
        )
        
        # Packaging features
        features_df['packaging_recyclability'] = features_df.get('packaging_recyclable', 0).astype(int)
        features_df['packaging_biodegradable'] = features_df.get('packaging_biodegradable', 0).astype(int)
        
        # Certification features
        certifications = ['organic', 'fair_trade', 'carbon_neutral', 'renewable_energy']
        for cert in certifications:
            features_df[f'has_{cert}'] = features_df.get(cert, 0).astype(int)
        
        # Supply chain features
        features_df['local_sourcing'] = (features_df.get('transportation_distance', 1000) < 500).astype(int)
        features_df['renewable_energy_use'] = features_df.get('renewable_energy_percentage', 0) / 100
        
        # Durability and lifespan features
        features_df['durability_score'] = features_df.get('expected_lifespan', 1) / features_df.get('industry_avg_lifespan', 1)
        features_df['repairability_score'] = features_df.get('repairability_index', 5) / 10
        
        # Social impact features
        features_df['labor_practices_score'] = features_df.get('labor_practices_rating', 3) / 5
        features_df['community_impact_score'] = features_df.get('community_impact_rating', 3) / 5
        
        return features_df
    
    def calculate_component_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate individual component scores for sustainability
        
        Args:
            df: DataFrame with prepared features
            
        Returns:
            DataFrame with component scores
        """
        scores_df = df.copy()
        
        # Emission score (0-100, higher is better)
        max_carbon = df['carbon_footprint'].quantile(0.95)
        scores_df['emission_score'] = np.clip(
            100 - (df['carbon_footprint'] / max_carbon * 100), 0, 100
        )
        
        # Resource usage score (water + energy)
        max_water = df['water_usage'].quantile(0.95)
        max_energy = df['energy_consumption'].quantile(0.95)
        
        water_score = np.clip(100 - (df['water_usage'] / max_water * 100), 0, 100)
        energy_score = np.clip(100 - (df['energy_consumption'] / max_energy * 100), 0, 100)
        scores_df['resource_usage_score'] = (water_score + energy_score) / 2
        
        # Transportation score
        max_transport = df['transportation_distance'].quantile(0.95)
        scores_df['transportation_score'] = np.clip(
            100 - (df['transportation_distance'] / max_transport * 100), 0, 100
        )
        
        # Recyclability score
        scores_df['recyclability_score'] = (
            df['packaging_recyclability'] * 40 +
            df['packaging_biodegradable'] * 30 +
            df['repairability_score'] * 30
        )
        
        # Labor practices score
        scores_df['labor_practices_score'] = (
            df['labor_practices_score'] * 60 +
            df['community_impact_score'] * 40
        )
        
        # Certification bonus
        cert_cols = [col for col in df.columns if col.startswith('has_')]
        scores_df['certification_bonus'] = df[cert_cols].sum(axis=1) * 5
        
        return scores_df
    
    def train_model(self, df: pd.DataFrame, target_column: str = 'sustainability_score') -> Dict[str, Any]:
        """
        Train the sustainability scoring model
        
        Args:
            df: Training data
            target_column: Name of the target column
            
        Returns:
            Training results and metrics
        """
        # Prepare features
        feature_df = self.prepare_features(df)
        
        # Calculate component scores if target doesn't exist
        if target_column not in df.columns:
            score_df = self.calculate_component_scores(feature_df)
            # Create composite sustainability score
            weights = {
                'emission_score': 0.25,
                'resource_usage_score': 0.20,
                'transportation_score': 0.15,
                'recyclability_score': 0.15,
                'labor_practices_score': 0.15,
                'certification_bonus': 0.10
            }
            
            df[target_column] = sum(
                score_df[score] * weight for score, weight in weights.items()
            )
        
        # Select features for training
        self.feature_columns = [
            'carbon_per_kg', 'carbon_per_dollar', 'energy_per_kg', 'energy_per_dollar',
            'water_per_kg', 'water_per_dollar', 'transport_emissions',
            'packaging_recyclability', 'packaging_biodegradable',
            'local_sourcing', 'renewable_energy_use', 'durability_score',
            'repairability_score', 'labor_practices_score', 'community_impact_score'
        ] + [col for col in feature_df.columns if col.startswith('has_')]
        
        # Filter available columns
        available_columns = [col for col in self.feature_columns if col in feature_df.columns]
        self.feature_columns = available_columns
        
        X = feature_df[self.feature_columns].fillna(0)
        y = df[target_column].fillna(df[target_column].median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
        
        results = {
            'model_type': self.model_type,
            'version': self.version,
            'mse': mse,
            'r2_score': r2,
            'feature_importance': self.feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"Model trained successfully. R2 Score: {r2:.4f}, MSE: {mse:.4f}")
        
        return results
    
    def predict_sustainability_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict sustainability scores for new products
        
        Args:
            df: DataFrame with product data
            
        Returns:
            Dictionary with scores and explanations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare features
        feature_df = self.prepare_features(df)
        
        # Select features
        X = feature_df[self.feature_columns].fillna(0)
        
        # Make predictions
        scores = self.model.predict(X)
        
        # Calculate component scores for explanation
        component_scores = self.calculate_component_scores(feature_df)
        
        # Generate explanations
        explanations = []
        for i, score in enumerate(scores):
            explanation = self._generate_explanation(
                feature_df.iloc[i], 
                component_scores.iloc[i], 
                score
            )
            explanations.append(explanation)
        
        return {
            'sustainability_scores': scores.tolist(),
            'component_scores': component_scores[
                ['emission_score', 'resource_usage_score', 'transportation_score',
                 'recyclability_score', 'labor_practices_score']
            ].to_dict('records'),
            'explanations': explanations,
            'model_version': self.version,
            'predicted_at': datetime.now().isoformat()
        }
    
    def _generate_explanation(self, features: pd.Series, component_scores: pd.Series, overall_score: float) -> Dict[str, Any]:
        """
        Generate explanation for sustainability score
        
        Args:
            features: Product features
            component_scores: Component scores
            overall_score: Overall sustainability score
            
        Returns:
            Explanation dictionary
        """
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Check each component
        if component_scores.get('emission_score', 0) > 70:
            strengths.append("Low carbon footprint")
        elif component_scores.get('emission_score', 0) < 30:
            weaknesses.append("High carbon emissions")
            recommendations.append("Consider renewable energy sources")
        
        if component_scores.get('resource_usage_score', 0) > 70:
            strengths.append("Efficient resource usage")
        elif component_scores.get('resource_usage_score', 0) < 30:
            weaknesses.append("High resource consumption")
            recommendations.append("Optimize water and energy usage")
        
        if component_scores.get('transportation_score', 0) > 70:
            strengths.append("Low transportation impact")
        elif component_scores.get('transportation_score', 0) < 30:
            weaknesses.append("High transportation emissions")
            recommendations.append("Source materials locally")
        
        if component_scores.get('recyclability_score', 0) > 70:
            strengths.append("Highly recyclable")
        elif component_scores.get('recyclability_score', 0) < 30:
            weaknesses.append("Limited recyclability")
            recommendations.append("Improve packaging design")
        
        if component_scores.get('labor_practices_score', 0) > 70:
            strengths.append("Good labor practices")
        elif component_scores.get('labor_practices_score', 0) < 30:
            weaknesses.append("Poor labor practices")
            recommendations.append("Improve working conditions")
        
        # Feature importance insights
        top_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'overall_score': overall_score,
            'grade': self._score_to_grade(overall_score),
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'top_influencing_factors': [factor[0] for factor in top_features],
            'component_breakdown': {
                'emissions': component_scores.get('emission_score', 0),
                'resource_usage': component_scores.get('resource_usage_score', 0),
                'transportation': component_scores.get('transportation_score', 0),
                'recyclability': component_scores.get('recyclability_score', 0),
                'labor_practices': component_scores.get('labor_practices_score', 0)
            }
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'version': self.version,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        self.version = model_data['version']
        
        logger.info(f"Model loaded from {filepath}")
    
    def batch_score_products(self, df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        """
        Score products in batches for better performance
        
        Args:
            df: DataFrame with product data
            batch_size: Number of products to process at once
            
        Returns:
            DataFrame with sustainability scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_results = self.predict_sustainability_score(batch)
            
            batch_df = pd.DataFrame({
                'product_id': batch.index,
                'sustainability_score': batch_results['sustainability_scores'],
                'emission_score': [cs['emissions'] for cs in batch_results['component_scores']],
                'resource_usage_score': [cs['resource_usage'] for cs in batch_results['component_scores']],
                'transportation_score': [cs['transportation'] for cs in batch_results['component_scores']],
                'recyclability_score': [cs['recyclability'] for cs in batch_results['component_scores']],
                'labor_practices_score': [cs['labor_practices'] for cs in batch_results['component_scores']],
                'grade': [exp['grade'] for exp in batch_results['explanations']]
            })
            
            results.append(batch_df)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
        
        return pd.concat(results, ignore_index=True)

class SustainabilityBenchmark:
    """Benchmarking and comparison tools for sustainability scores"""
    
    def __init__(self):
        self.industry_benchmarks = {}
        self.category_benchmarks = {}
    
    def create_industry_benchmarks(self, df: pd.DataFrame):
        """Create industry-specific sustainability benchmarks"""
        if 'industry' in df.columns and 'sustainability_score' in df.columns:
            self.industry_benchmarks = df.groupby('industry')['sustainability_score'].agg([
                'mean', 'median', 'std', 'count',
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
                lambda x: x.quantile(0.90)
            ]).to_dict()
    
    def create_category_benchmarks(self, df: pd.DataFrame):
        """Create category-specific sustainability benchmarks"""
        if 'category' in df.columns and 'sustainability_score' in df.columns:
            self.category_benchmarks = df.groupby('category')['sustainability_score'].agg([
                'mean', 'median', 'std', 'count',
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
                lambda x: x.quantile(0.90)
            ]).to_dict()
    
    def compare_to_benchmark(self, product_score: float, industry: str = None, category: str = None) -> Dict[str, Any]:
        """
        Compare product score to industry/category benchmarks
        
        Args:
            product_score: Product's sustainability score
            industry: Industry name
            category: Category name
            
        Returns:
            Comparison results
        """
        comparison = {
            'score': product_score,
            'industry_comparison': None,
            'category_comparison': None
        }
        
        if industry and industry in self.industry_benchmarks:
            industry_mean = self.industry_benchmarks[industry]['mean']
            industry_median = self.industry_benchmarks[industry]['median']
            
            comparison['industry_comparison'] = {
                'industry': industry,
                'industry_mean': industry_mean,
                'industry_median': industry_median,
                'percentile_rank': self._calculate_percentile_rank(product_score, industry, 'industry'),
                'performance': 'Above Average' if product_score > industry_mean else 'Below Average'
            }
        
        if category and category in self.category_benchmarks:
            category_mean = self.category_benchmarks[category]['mean']
            category_median = self.category_benchmarks[category]['median']
            
            comparison['category_comparison'] = {
                'category': category,
                'category_mean': category_mean,
                'category_median': category_median,
                'percentile_rank': self._calculate_percentile_rank(product_score, category, 'category'),
                'performance': 'Above Average' if product_score > category_mean else 'Below Average'
            }
        
        return comparison
    
    def _calculate_percentile_rank(self, score: float, group: str, group_type: str) -> float:
        """Calculate percentile rank within group"""
        # Placeholder implementation
        # In practice, you'd need historical data to calculate this accurately
        return 50.0
