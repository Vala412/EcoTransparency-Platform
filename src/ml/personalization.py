"""
Personalization Engine for Sustainable Product Recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class PersonalizationEngine:
    """Main personalization engine for sustainable product recommendations"""
    
    def __init__(self):
        self.collaborative_filter = CollaborativeFilter()
        self.content_filter = ContentBasedFilter()
        self.hybrid_recommender = HybridRecommender()
        self.user_profiler = UserProfiler()
        self.sustainability_matcher = SustainabilityMatcher()
        
        # Model weights for hybrid approach
        self.model_weights = {
            'collaborative': 0.4,
            'content': 0.3,
            'sustainability': 0.3
        }
    
    def create_user_profile(self, user_id: str, interaction_data: pd.DataFrame,
                          preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create comprehensive user profile
        
        Args:
            user_id: User identifier
            interaction_data: User's interaction history
            preferences: Explicit user preferences
            
        Returns:
            User profile dictionary
        """
        profile = self.user_profiler.create_profile(user_id, interaction_data, preferences)
        
        # Add sustainability preferences
        profile['sustainability_profile'] = self.sustainability_matcher.analyze_user_preferences(
            interaction_data, profile
        )
        
        return profile
    
    def generate_recommendations(self, user_id: str, user_profile: Dict[str, Any],
                               products_df: pd.DataFrame, num_recommendations: int = 10,
                               filter_purchased: bool = True) -> List[Dict[str, Any]]:
        """
        Generate personalized product recommendations
        
        Args:
            user_id: User identifier
            user_profile: User profile dictionary
            products_df: DataFrame with product information
            num_recommendations: Number of recommendations to generate
            filter_purchased: Whether to filter out already purchased products
            
        Returns:
            List of recommended products with scores
        """
        recommendations = []
        
        # Get collaborative filtering recommendations
        collab_recs = self.collaborative_filter.get_recommendations(
            user_id, products_df, num_recommendations * 2
        )
        
        # Get content-based recommendations
        content_recs = self.content_filter.get_recommendations(
            user_profile, products_df, num_recommendations * 2
        )
        
        # Get sustainability-based recommendations
        sustainability_recs = self.sustainability_matcher.get_recommendations(
            user_profile['sustainability_profile'], products_df, num_recommendations * 2
        )
        
        # Combine using hybrid approach
        combined_recs = self.hybrid_recommender.combine_recommendations(
            collab_recs, content_recs, sustainability_recs, self.model_weights
        )
        
        # Filter and rank recommendations
        final_recs = self._filter_and_rank_recommendations(
            combined_recs, user_profile, products_df, num_recommendations, filter_purchased
        )
        
        # Add explanations
        for rec in final_recs:
            rec['explanation'] = self._generate_explanation(rec, user_profile)
        
        return final_recs
    
    def _filter_and_rank_recommendations(self, recommendations: List[Dict[str, Any]],
                                       user_profile: Dict[str, Any], products_df: pd.DataFrame,
                                       num_recommendations: int, filter_purchased: bool) -> List[Dict[str, Any]]:
        """Filter and rank recommendations based on user preferences"""
        
        # Filter out purchased products if requested
        if filter_purchased and 'purchased_products' in user_profile:
            recommendations = [
                rec for rec in recommendations 
                if rec['product_id'] not in user_profile['purchased_products']
            ]
        
        # Apply preference filters
        filtered_recs = []
        for rec in recommendations:
            product_info = products_df[products_df['id'] == rec['product_id']].iloc[0]
            
            # Check price preference
            if 'price_range' in user_profile and user_profile['price_range']:
                price_min, price_max = user_profile['price_range']
                if not (price_min <= product_info.get('price', 0) <= price_max):
                    continue
            
            # Check category preferences
            if 'preferred_categories' in user_profile and user_profile['preferred_categories']:
                if product_info.get('category') not in user_profile['preferred_categories']:
                    continue
            
            # Check sustainability requirements
            if 'min_sustainability_score' in user_profile:
                if product_info.get('sustainability_score', 0) < user_profile['min_sustainability_score']:
                    continue
            
            filtered_recs.append(rec)
        
        # Sort by combined score
        filtered_recs.sort(key=lambda x: x['score'], reverse=True)
        
        return filtered_recs[:num_recommendations]
    
    def _generate_explanation(self, recommendation: Dict[str, Any], 
                            user_profile: Dict[str, Any]) -> str:
        """Generate explanation for why product was recommended"""
        reasons = []
        
        # Sustainability reasons
        if recommendation.get('sustainability_score', 0) > 0.7:
            reasons.append("high sustainability score")
        
        # Category preference
        if 'preferred_categories' in user_profile:
            if recommendation.get('category') in user_profile['preferred_categories']:
                reasons.append("matches your preferred categories")
        
        # Similar users
        if recommendation.get('collaborative_score', 0) > 0.5:
            reasons.append("popular among similar users")
        
        # Content similarity
        if recommendation.get('content_score', 0) > 0.5:
            reasons.append("similar to your previous purchases")
        
        if not reasons:
            reasons.append("personalized for you")
        
        return f"Recommended because it has {', '.join(reasons)}"
    
    def update_user_feedback(self, user_id: str, product_id: str, feedback_type: str,
                           rating: float = None, implicit_feedback: bool = False):
        """
        Update user model based on feedback
        
        Args:
            user_id: User identifier
            product_id: Product identifier
            feedback_type: Type of feedback ('like', 'dislike', 'purchase', 'view')
            rating: Explicit rating (1-5)
            implicit_feedback: Whether feedback is implicit
        """
        feedback_data = {
            'user_id': user_id,
            'product_id': product_id,
            'feedback_type': feedback_type,
            'rating': rating,
            'implicit': implicit_feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update collaborative filter
        self.collaborative_filter.update_feedback(feedback_data)
        
        # Update content filter
        self.content_filter.update_feedback(feedback_data)
        
        # Update sustainability matcher
        self.sustainability_matcher.update_feedback(feedback_data)
    
    def get_trending_sustainable_products(self, products_df: pd.DataFrame,
                                        time_window: int = 7) -> List[Dict[str, Any]]:
        """Get trending sustainable products"""
        # Filter for high sustainability scores
        sustainable_products = products_df[products_df['sustainability_score'] >= 70]
        
        # Mock trending logic (in practice, would use actual interaction data)
        trending_products = sustainable_products.sample(n=min(10, len(sustainable_products)))
        
        return [
            {
                'product_id': row['id'],
                'name': row['name'],
                'sustainability_score': row['sustainability_score'],
                'trend_score': np.random.uniform(0.7, 1.0)  # Mock trend score
            }
            for _, row in trending_products.iterrows()
        ]

class UserProfiler:
    """Create and manage user profiles"""
    
    def __init__(self):
        self.profiles = {}
    
    def create_profile(self, user_id: str, interaction_data: pd.DataFrame,
                      preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create user profile from interaction data and preferences"""
        
        profile = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'interaction_count': len(interaction_data),
            'preferences': preferences or {},
            'behavioral_features': {},
            'sustainability_preferences': {},
            'category_preferences': {},
            'price_sensitivity': 0.5,
            'purchase_history': [],
            'preferred_categories': [],
            'price_range': None
        }
        
        if not interaction_data.empty:
            # Analyze behavioral patterns
            profile['behavioral_features'] = self._analyze_behavior(interaction_data)
            
            # Extract category preferences
            profile['category_preferences'] = self._extract_category_preferences(interaction_data)
            
            # Calculate price sensitivity
            profile['price_sensitivity'] = self._calculate_price_sensitivity(interaction_data)
            
            # Extract preferred categories
            profile['preferred_categories'] = self._get_preferred_categories(interaction_data)
            
            # Calculate price range
            profile['price_range'] = self._calculate_price_range(interaction_data)
            
            # Extract purchase history
            profile['purchase_history'] = self._extract_purchase_history(interaction_data)
        
        # Add explicit preferences
        if preferences:
            profile['preferences'].update(preferences)
            
            # Extract sustainability preferences
            if 'sustainability_priorities' in preferences:
                profile['sustainability_preferences'] = preferences['sustainability_priorities']
            
            if 'min_sustainability_score' in preferences:
                profile['min_sustainability_score'] = preferences['min_sustainability_score']
        
        self.profiles[user_id] = profile
        return profile
    
    def _analyze_behavior(self, interaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavioral patterns"""
        behavior = {
            'avg_session_duration': 0,
            'purchase_frequency': 0,
            'browsing_patterns': {},
            'seasonal_patterns': {},
            'time_of_day_patterns': {}
        }
        
        if 'interaction_type' in interaction_data.columns:
            # Purchase frequency
            purchases = interaction_data[interaction_data['interaction_type'] == 'purchase']
            behavior['purchase_frequency'] = len(purchases) / len(interaction_data)
            
            # Browsing patterns
            behavior['browsing_patterns'] = interaction_data['interaction_type'].value_counts().to_dict()
        
        if 'timestamp' in interaction_data.columns:
            interaction_data['timestamp'] = pd.to_datetime(interaction_data['timestamp'])
            
            # Time of day patterns
            interaction_data['hour'] = interaction_data['timestamp'].dt.hour
            behavior['time_of_day_patterns'] = interaction_data.groupby('hour').size().to_dict()
            
            # Seasonal patterns
            interaction_data['month'] = interaction_data['timestamp'].dt.month
            behavior['seasonal_patterns'] = interaction_data.groupby('month').size().to_dict()
        
        return behavior
    
    def _extract_category_preferences(self, interaction_data: pd.DataFrame) -> Dict[str, float]:
        """Extract category preferences from interaction data"""
        if 'category' not in interaction_data.columns:
            return {}
        
        # Weight different interaction types
        interaction_weights = {
            'purchase': 1.0,
            'like': 0.8,
            'view': 0.3,
            'dislike': -0.5
        }
        
        category_scores = defaultdict(float)
        
        for _, row in interaction_data.iterrows():
            category = row['category']
            interaction_type = row.get('interaction_type', 'view')
            weight = interaction_weights.get(interaction_type, 0.1)
            
            category_scores[category] += weight
        
        # Normalize scores
        max_score = max(category_scores.values()) if category_scores else 1
        return {cat: score / max_score for cat, score in category_scores.items()}
    
    def _calculate_price_sensitivity(self, interaction_data: pd.DataFrame) -> float:
        """Calculate user's price sensitivity"""
        if 'price' not in interaction_data.columns:
            return 0.5
        
        # Simple price sensitivity calculation
        purchases = interaction_data[interaction_data.get('interaction_type') == 'purchase']
        
        if len(purchases) == 0:
            return 0.5
        
        avg_purchase_price = purchases['price'].mean()
        max_price = interaction_data['price'].max()
        
        # Normalize to 0-1 scale (lower values = more price sensitive)
        return 1 - (avg_purchase_price / max_price)
    
    def _get_preferred_categories(self, interaction_data: pd.DataFrame) -> List[str]:
        """Get user's preferred categories"""
        if 'category' not in interaction_data.columns:
            return []
        
        category_counts = interaction_data['category'].value_counts()
        return category_counts.head(5).index.tolist()
    
    def _calculate_price_range(self, interaction_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate user's typical price range"""
        if 'price' not in interaction_data.columns:
            return None
        
        prices = interaction_data['price'].dropna()
        if len(prices) == 0:
            return None
        
        return (prices.quantile(0.1), prices.quantile(0.9))
    
    def _extract_purchase_history(self, interaction_data: pd.DataFrame) -> List[str]:
        """Extract user's purchase history"""
        if 'interaction_type' not in interaction_data.columns:
            return []
        
        purchases = interaction_data[interaction_data['interaction_type'] == 'purchase']
        return purchases['product_id'].tolist()
    
    def update_profile(self, user_id: str, new_interaction_data: pd.DataFrame):
        """Update user profile with new interaction data"""
        if user_id not in self.profiles:
            return
        
        profile = self.profiles[user_id]
        
        # Update interaction count
        profile['interaction_count'] += len(new_interaction_data)
        
        # Update behavioral features
        new_behavior = self._analyze_behavior(new_interaction_data)
        profile['behavioral_features'].update(new_behavior)
        
        # Update category preferences
        new_category_prefs = self._extract_category_preferences(new_interaction_data)
        for cat, score in new_category_prefs.items():
            if cat in profile['category_preferences']:
                profile['category_preferences'][cat] = (
                    profile['category_preferences'][cat] * 0.8 + score * 0.2
                )
            else:
                profile['category_preferences'][cat] = score
        
        # Update purchase history
        new_purchases = self._extract_purchase_history(new_interaction_data)
        profile['purchase_history'].extend(new_purchases)
        
        profile['updated_at'] = datetime.now().isoformat()

class CollaborativeFilter:
    """Collaborative filtering recommendation system"""
    
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.svd_model = None
        self.user_encoders = {}
        self.item_encoders = {}
    
    def fit(self, interaction_data: pd.DataFrame):
        """Fit collaborative filtering model"""
        # Create user-item interaction matrix
        self.user_item_matrix = interaction_data.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )
        
        # Use SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        user_features = self.svd_model.fit_transform(self.user_item_matrix)
        
        # Calculate item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # Create encoders for new users/items
        self.user_encoders = {user: idx for idx, user in enumerate(self.user_item_matrix.index)}
        self.item_encoders = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
    
    def get_recommendations(self, user_id: str, products_df: pd.DataFrame,
                          num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations"""
        if self.user_item_matrix is None:
            return []
        
        if user_id not in self.user_encoders:
            return []  # Cold start problem
        
        user_idx = self.user_encoders[user_id]
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Find similar users
        user_similarity = cosine_similarity([user_ratings], self.user_item_matrix)[0]
        similar_users = np.argsort(user_similarity)[-11:-1]  # Top 10 similar users
        
        # Get recommendations based on similar users
        recommendations = []
        for item_idx, item_id in enumerate(self.user_item_matrix.columns):
            if user_ratings.iloc[item_idx] > 0:  # Skip items user has already rated
                continue
            
            # Calculate predicted rating
            similar_user_ratings = self.user_item_matrix.iloc[similar_users, item_idx]
            similar_user_similarities = user_similarity[similar_users]
            
            # Weighted average of similar users' ratings
            if np.sum(similar_user_similarities) > 0:
                predicted_rating = np.sum(similar_user_ratings * similar_user_similarities) / np.sum(similar_user_similarities)
            else:
                predicted_rating = 0
            
            if predicted_rating > 0:
                recommendations.append({
                    'product_id': item_id,
                    'score': predicted_rating,
                    'collaborative_score': predicted_rating
                })
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:num_recommendations]
    
    def update_feedback(self, feedback_data: Dict[str, Any]):
        """Update model with new feedback"""
        # In practice, this would update the user-item matrix
        # and potentially retrain the model
        pass

class ContentBasedFilter:
    """Content-based filtering recommendation system"""
    
    def __init__(self):
        self.product_features = None
        self.feature_scaler = StandardScaler()
        self.product_similarity_matrix = None
    
    def fit(self, products_df: pd.DataFrame):
        """Fit content-based filtering model"""
        # Extract product features
        feature_columns = [
            'sustainability_score', 'carbon_footprint', 'water_usage',
            'energy_consumption', 'price', 'recyclability_score'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in products_df.columns]
        
        if not available_features:
            return
        
        self.product_features = products_df[['id'] + available_features].copy()
        
        # Handle missing values
        self.product_features[available_features] = self.product_features[available_features].fillna(0)
        
        # Scale features
        feature_matrix = self.feature_scaler.fit_transform(self.product_features[available_features])
        
        # Calculate product similarity matrix
        self.product_similarity_matrix = cosine_similarity(feature_matrix)
    
    def get_recommendations(self, user_profile: Dict[str, Any], products_df: pd.DataFrame,
                          num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get content-based recommendations"""
        if self.product_features is None:
            return []
        
        recommendations = []
        
        # Get user's preferred product characteristics
        purchase_history = user_profile.get('purchase_history', [])
        
        if not purchase_history:
            return []
        
        # Calculate average characteristics of purchased products
        purchased_products = self.product_features[
            self.product_features['id'].isin(purchase_history)
        ]
        
        if purchased_products.empty:
            return []
        
        # Calculate similarity to user's preferences
        user_feature_profile = purchased_products.select_dtypes(include=[np.number]).mean()
        
        for idx, row in self.product_features.iterrows():
            if row['id'] in purchase_history:
                continue
            
            # Calculate similarity
            product_features = row.select_dtypes(include=[np.number])
            similarity = cosine_similarity(
                [user_feature_profile.values],
                [product_features.values]
            )[0][0]
            
            recommendations.append({
                'product_id': row['id'],
                'score': similarity,
                'content_score': similarity
            })
        
        # Sort by similarity and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:num_recommendations]
    
    def update_feedback(self, feedback_data: Dict[str, Any]):
        """Update model with new feedback"""
        # In practice, this would update user preferences
        pass

class SustainabilityMatcher:
    """Sustainability-based recommendation system"""
    
    def __init__(self):
        self.sustainability_weights = {
            'low_carbon': 0.3,
            'plastic_free': 0.2,
            'local': 0.2,
            'renewable_energy': 0.15,
            'fair_trade': 0.15
        }
    
    def analyze_user_preferences(self, interaction_data: pd.DataFrame,
                               user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's sustainability preferences"""
        sustainability_profile = {
            'priorities': [],
            'min_score_threshold': 50,
            'preferred_certifications': [],
            'eco_friendly_tendency': 0.5
        }
        
        # Extract from explicit preferences
        if 'sustainability_priorities' in user_profile.get('preferences', {}):
            sustainability_profile['priorities'] = user_profile['preferences']['sustainability_priorities']
        
        # Analyze from interaction data
        if 'sustainability_score' in interaction_data.columns:
            avg_sustainability = interaction_data['sustainability_score'].mean()
            sustainability_profile['eco_friendly_tendency'] = avg_sustainability / 100
            sustainability_profile['min_score_threshold'] = max(50, avg_sustainability - 20)
        
        return sustainability_profile
    
    def get_recommendations(self, sustainability_profile: Dict[str, Any],
                          products_df: pd.DataFrame, num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get sustainability-based recommendations"""
        recommendations = []
        
        # Filter products by minimum sustainability score
        min_score = sustainability_profile.get('min_score_threshold', 50)
        sustainable_products = products_df[
            products_df.get('sustainability_score', 0) >= min_score
        ]
        
        for _, product in sustainable_products.iterrows():
            score = self._calculate_sustainability_score(product, sustainability_profile)
            
            recommendations.append({
                'product_id': product['id'],
                'score': score,
                'sustainability_score': score,
                'environmental_impact': product.get('sustainability_score', 0)
            })
        
        # Sort by sustainability score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:num_recommendations]
    
    def _calculate_sustainability_score(self, product: pd.Series,
                                      sustainability_profile: Dict[str, Any]) -> float:
        """Calculate sustainability score for a product"""
        base_score = product.get('sustainability_score', 0) / 100
        
        # Boost score based on user priorities
        priorities = sustainability_profile.get('priorities', [])
        
        bonus = 0
        for priority in priorities:
            if priority == 'low_carbon' and product.get('carbon_footprint', 0) < 10:
                bonus += 0.2
            elif priority == 'local' and product.get('transportation_distance', 1000) < 500:
                bonus += 0.2
            elif priority == 'renewable_energy' and product.get('renewable_energy_percentage', 0) > 80:
                bonus += 0.2
        
        return min(base_score + bonus, 1.0)
    
    def update_feedback(self, feedback_data: Dict[str, Any]):
        """Update sustainability preferences based on feedback"""
        # In practice, this would update user sustainability preferences
        pass

class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches"""
    
    def combine_recommendations(self, collab_recs: List[Dict[str, Any]],
                              content_recs: List[Dict[str, Any]],
                              sustainability_recs: List[Dict[str, Any]],
                              weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Combine recommendations from multiple systems"""
        
        # Create product score dictionary
        product_scores = defaultdict(lambda: {
            'collaborative': 0,
            'content': 0,
            'sustainability': 0,
            'count': 0
        })
        
        # Add collaborative scores
        for rec in collab_recs:
            product_id = rec['product_id']
            product_scores[product_id]['collaborative'] = rec['score']
            product_scores[product_id]['count'] += 1
        
        # Add content scores
        for rec in content_recs:
            product_id = rec['product_id']
            product_scores[product_id]['content'] = rec['score']
            product_scores[product_id]['count'] += 1
        
        # Add sustainability scores
        for rec in sustainability_recs:
            product_id = rec['product_id']
            product_scores[product_id]['sustainability'] = rec['score']
            product_scores[product_id]['count'] += 1
        
        # Calculate combined scores
        combined_recommendations = []
        
        for product_id, scores in product_scores.items():
            if scores['count'] == 0:
                continue
            
            # Calculate weighted average
            combined_score = (
                scores['collaborative'] * weights['collaborative'] +
                scores['content'] * weights['content'] +
                scores['sustainability'] * weights['sustainability']
            )
            
            combined_recommendations.append({
                'product_id': product_id,
                'score': combined_score,
                'collaborative_score': scores['collaborative'],
                'content_score': scores['content'],
                'sustainability_score': scores['sustainability'],
                'diversity_bonus': 0.1 if scores['count'] > 1 else 0
            })
        
        # Sort by combined score
        combined_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_recommendations
