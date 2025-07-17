"""
Greenwashing Detection Module using Computer Vision and NLP
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import cv2
import pytesseract
from PIL import Image
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime
import spacy
from collections import Counter

logger = logging.getLogger(__name__)

class GreenwashingDetector:
    """Main greenwashing detection system"""
    
    def __init__(self):
        self.text_analyzer = GreenwashingTextAnalyzer()
        self.image_analyzer = GreenwashingImageAnalyzer()
        self.certification_validator = CertificationValidator()
        self.combined_model = None
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def analyze_product_claims(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze product for greenwashing indicators
        
        Args:
            product_data: Dictionary containing product information
            
        Returns:
            Greenwashing analysis results
        """
        results = {
            'product_id': product_data.get('id'),
            'overall_risk_score': 0.0,
            'risk_level': 'low',
            'alerts': [],
            'text_analysis': None,
            'image_analysis': None,
            'certification_analysis': None,
            'recommendations': []
        }
        
        # Analyze text claims
        if 'description' in product_data or 'marketing_text' in product_data:
            text_content = product_data.get('description', '') + ' ' + product_data.get('marketing_text', '')
            results['text_analysis'] = self.text_analyzer.analyze_text_claims(text_content)
        
        # Analyze images
        if 'image_paths' in product_data:
            results['image_analysis'] = self.image_analyzer.analyze_product_images(product_data['image_paths'])
        
        # Validate certifications
        if 'certifications' in product_data:
            results['certification_analysis'] = self.certification_validator.validate_certifications(
                product_data['certifications']
            )
        
        # Calculate overall risk score
        risk_components = []
        if results['text_analysis']:
            risk_components.append(results['text_analysis']['risk_score'])
        if results['image_analysis']:
            risk_components.append(results['image_analysis']['risk_score'])
        if results['certification_analysis']:
            risk_components.append(results['certification_analysis']['risk_score'])
        
        if risk_components:
            results['overall_risk_score'] = np.mean(risk_components)
            results['risk_level'] = self._classify_risk_level(results['overall_risk_score'])
        
        # Generate alerts and recommendations
        results['alerts'] = self._generate_alerts(results)
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score"""
        if risk_score >= self.risk_thresholds['high']:
            return 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_alerts(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on analysis results"""
        alerts = []
        
        # Text-based alerts
        if results.get('text_analysis'):
            text_analysis = results['text_analysis']
            if text_analysis['risk_score'] > self.risk_thresholds['medium']:
                alerts.append({
                    'type': 'text_greenwashing',
                    'severity': 'medium' if text_analysis['risk_score'] < self.risk_thresholds['high'] else 'high',
                    'message': f"Suspicious text patterns detected: {', '.join(text_analysis['red_flags'])}",
                    'evidence': text_analysis['flagged_phrases']
                })
        
        # Image-based alerts
        if results.get('image_analysis'):
            image_analysis = results['image_analysis']
            if image_analysis['risk_score'] > self.risk_thresholds['medium']:
                alerts.append({
                    'type': 'image_greenwashing',
                    'severity': 'medium' if image_analysis['risk_score'] < self.risk_thresholds['high'] else 'high',
                    'message': "Suspicious imagery or claims detected in product images",
                    'evidence': image_analysis['flagged_elements']
                })
        
        # Certification alerts
        if results.get('certification_analysis'):
            cert_analysis = results['certification_analysis']
            if cert_analysis['invalid_certifications']:
                alerts.append({
                    'type': 'invalid_certification',
                    'severity': 'high',
                    'message': "Invalid or unverified certifications found",
                    'evidence': cert_analysis['invalid_certifications']
                })
        
        return alerts
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if results['risk_level'] == 'high':
            recommendations.append("Immediate review required - high risk of greenwashing")
            recommendations.append("Verify all environmental claims with supporting data")
        
        if results.get('text_analysis', {}).get('lacks_specificity'):
            recommendations.append("Provide specific, quantifiable environmental data")
        
        if results.get('certification_analysis', {}).get('invalid_certifications'):
            recommendations.append("Remove invalid certifications and obtain proper accreditation")
        
        if results.get('image_analysis', {}).get('misleading_imagery'):
            recommendations.append("Review product imagery for misleading environmental representations")
        
        return recommendations

class GreenwashingTextAnalyzer:
    """NLP-based text analysis for greenwashing detection"""
    
    def __init__(self):
        self.nlp = None
        self.greenwashing_patterns = self._load_greenwashing_patterns()
        self.environmental_keywords = self._load_environmental_keywords()
        self.vague_terms = self._load_vague_terms()
        self.model = None
        self.vectorizer = None
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    
    def _load_greenwashing_patterns(self) -> Dict[str, List[str]]:
        """Load patterns associated with greenwashing"""
        return {
            'vague_claims': [
                r'eco-friendly', r'green', r'natural', r'sustainable', r'environmentally safe',
                r'eco-conscious', r'earth-friendly', r'eco-safe', r'environmentally responsible'
            ],
            'absolute_claims': [
                r'100% natural', r'completely safe', r'totally eco-friendly', r'entirely sustainable',
                r'perfectly green', r'absolutely clean'
            ],
            'misleading_terms': [
                r'chemical-free', r'toxin-free', r'non-toxic', r'safe for environment',
                r'environmentally neutral', r'carbon-free'
            ],
            'unsupported_claims': [
                r'scientifically proven', r'studies show', r'research indicates',
                r'experts recommend', r'laboratory tested'
            ]
        }
    
    def _load_environmental_keywords(self) -> List[str]:
        """Load legitimate environmental keywords"""
        return [
            'carbon footprint', 'renewable energy', 'biodegradable', 'recyclable',
            'organic certified', 'fair trade', 'carbon neutral', 'zero waste',
            'solar powered', 'wind energy', 'compostable', 'cruelty-free',
            'sustainable sourcing', 'life cycle assessment', 'environmental impact'
        ]
    
    def _load_vague_terms(self) -> List[str]:
        """Load vague terms that lack specificity"""
        return [
            'eco-friendly', 'green', 'natural', 'clean', 'pure', 'fresh',
            'healthy', 'safe', 'gentle', 'mild', 'non-toxic', 'chemical-free'
        ]
    
    def analyze_text_claims(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for greenwashing indicators
        
        Args:
            text: Text content to analyze
            
        Returns:
            Analysis results
        """
        analysis = {
            'risk_score': 0.0,
            'red_flags': [],
            'flagged_phrases': [],
            'vague_term_count': 0,
            'specific_claim_count': 0,
            'lacks_specificity': False,
            'unsupported_claims': [],
            'sentiment_score': 0.0,
            'keyword_density': {}
        }
        
        if not text:
            return analysis
        
        text_lower = text.lower()
        
        # Check for greenwashing patterns
        risk_score = 0.0
        
        for pattern_type, patterns in self.greenwashing_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    analysis['red_flags'].append(pattern_type)
                    analysis['flagged_phrases'].extend(matches)
                    
                    # Assign risk scores based on pattern type
                    if pattern_type == 'vague_claims':
                        risk_score += len(matches) * 0.3
                    elif pattern_type == 'absolute_claims':
                        risk_score += len(matches) * 0.5
                    elif pattern_type == 'misleading_terms':
                        risk_score += len(matches) * 0.4
                    elif pattern_type == 'unsupported_claims':
                        risk_score += len(matches) * 0.6
        
        # Count vague terms
        vague_count = sum(1 for term in self.vague_terms if term in text_lower)
        analysis['vague_term_count'] = vague_count
        
        # Count specific environmental claims
        specific_count = sum(1 for keyword in self.environmental_keywords if keyword in text_lower)
        analysis['specific_claim_count'] = specific_count
        
        # Check for lack of specificity
        if vague_count > specific_count * 2:
            analysis['lacks_specificity'] = True
            risk_score += 0.3
        
        # Calculate keyword density
        words = text_lower.split()
        total_words = len(words)
        
        if total_words > 0:
            env_keywords_found = [word for word in words if word in self.environmental_keywords]
            analysis['keyword_density'] = {
                'environmental_keywords': len(env_keywords_found) / total_words,
                'vague_terms': vague_count / total_words
            }
        
        # Sentiment analysis (simplified)
        analysis['sentiment_score'] = self._analyze_sentiment(text)
        
        # Normalize risk score
        analysis['risk_score'] = min(risk_score, 1.0)
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        if not self.nlp:
            return 0.0
        
        # Simplified sentiment scoring
        positive_indicators = ['good', 'great', 'excellent', 'amazing', 'perfect', 'best']
        negative_indicators = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'harmful']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_indicators if word in text_lower)
        negative_score = sum(1 for word in negative_indicators if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        return (positive_score - negative_score) / total_words
    
    def train_classification_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train a classification model for greenwashing detection
        
        Args:
            training_data: List of dictionaries with 'text' and 'is_greenwashing' fields
            
        Returns:
            Training results
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Prepare data
        texts = [item['text'] for item in training_data]
        labels = [item['is_greenwashing'] for item in training_data]
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        results = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'trained_at': datetime.now().isoformat()
        }
        
        return results

class GreenwashingImageAnalyzer:
    """Computer vision-based image analysis for greenwashing detection"""
    
    def __init__(self):
        self.certification_logos = self._load_certification_logos()
        self.misleading_imagery_patterns = self._load_misleading_patterns()
    
    def _load_certification_logos(self) -> Dict[str, str]:
        """Load known certification logos"""
        return {
            'organic_usda': 'USDA Organic',
            'fair_trade': 'Fair Trade Certified',
            'energy_star': 'Energy Star',
            'forest_stewardship': 'FSC Certified',
            'cradle_to_cradle': 'Cradle to Cradle',
            'carbon_trust': 'Carbon Trust',
            'rainforest_alliance': 'Rainforest Alliance'
        }
    
    def _load_misleading_patterns(self) -> List[str]:
        """Load patterns that indicate misleading imagery"""
        return [
            'excessive_green_color',
            'nature_imagery_unrelated',
            'false_certification_symbols',
            'misleading_recycling_symbols',
            'fake_natural_backgrounds'
        ]
    
    def analyze_product_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze product images for greenwashing indicators
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Image analysis results
        """
        analysis = {
            'risk_score': 0.0,
            'total_images': len(image_paths),
            'flagged_images': [],
            'certification_analysis': {},
            'text_extraction': {},
            'misleading_elements': [],
            'flagged_elements': []
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                image_analysis = self._analyze_single_image(image_path)
                
                if image_analysis['risk_score'] > 0.5:
                    analysis['flagged_images'].append({
                        'image_path': image_path,
                        'risk_score': image_analysis['risk_score'],
                        'issues': image_analysis['issues']
                    })
                
                # Aggregate results
                if image_analysis['extracted_text']:
                    analysis['text_extraction'][f'image_{i}'] = image_analysis['extracted_text']
                
                if image_analysis['certification_findings']:
                    analysis['certification_analysis'][f'image_{i}'] = image_analysis['certification_findings']
                
                if image_analysis['misleading_elements']:
                    analysis['misleading_elements'].extend(image_analysis['misleading_elements'])
                
            except Exception as e:
                logger.error(f"Error analyzing image {image_path}: {str(e)}")
                continue
        
        # Calculate overall risk score
        if analysis['flagged_images']:
            analysis['risk_score'] = np.mean([img['risk_score'] for img in analysis['flagged_images']])
        
        # Consolidate flagged elements
        analysis['flagged_elements'] = list(set(analysis['misleading_elements']))
        
        return analysis
    
    def _analyze_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single image for greenwashing indicators
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Single image analysis results
        """
        analysis = {
            'risk_score': 0.0,
            'issues': [],
            'extracted_text': '',
            'certification_findings': {},
            'misleading_elements': []
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                analysis['issues'].append('Could not load image')
                return analysis
            
            # Extract text using OCR
            analysis['extracted_text'] = self._extract_text_from_image(image)
            
            # Analyze color composition
            green_dominance = self._analyze_color_composition(image)
            if green_dominance > 0.4:  # More than 40% green
                analysis['risk_score'] += 0.3
                analysis['issues'].append('Excessive green color usage')
                analysis['misleading_elements'].append('excessive_green_color')
            
            # Check for certification logos
            cert_findings = self._detect_certification_logos(image)
            analysis['certification_findings'] = cert_findings
            
            # Check for misleading elements
            misleading_score = self._detect_misleading_elements(image, analysis['extracted_text'])
            analysis['risk_score'] += misleading_score
            
        except Exception as e:
            analysis['issues'].append(f'Analysis error: {str(e)}')
        
        return analysis
    
    def _extract_text_from_image(self, image: np.ndarray) -> str:
        """Extract text from image using OCR"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            text = pytesseract.image_to_string(thresh)
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return ""
    
    def _analyze_color_composition(self, image: np.ndarray) -> float:
        """Analyze color composition to detect excessive green usage"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green color range in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green pixels
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        green_pixels = np.sum(green_mask > 0)
        total_pixels = image.shape[0] * image.shape[1]
        
        return green_pixels / total_pixels
    
    def _detect_certification_logos(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect certification logos in image"""
        # Placeholder implementation
        # In practice, you would use template matching or trained models
        return {
            'detected_logos': [],
            'confidence_scores': {},
            'verified_certifications': []
        }
    
    def _detect_misleading_elements(self, image: np.ndarray, extracted_text: str) -> float:
        """Detect misleading visual elements"""
        misleading_score = 0.0
        
        # Check extracted text for misleading claims
        if extracted_text:
            text_lower = extracted_text.lower()
            misleading_terms = ['100% natural', 'completely eco-friendly', 'totally green', 'chemical-free']
            
            for term in misleading_terms:
                if term in text_lower:
                    misleading_score += 0.2
        
        # Additional computer vision checks would go here
        # (template matching, object detection, etc.)
        
        return min(misleading_score, 1.0)

class CertificationValidator:
    """Validate environmental certifications"""
    
    def __init__(self):
        self.valid_certifications = self._load_valid_certifications()
        self.certification_requirements = self._load_certification_requirements()
    
    def _load_valid_certifications(self) -> Dict[str, Dict[str, Any]]:
        """Load database of valid certifications"""
        return {
            'USDA Organic': {
                'issuer': 'United States Department of Agriculture',
                'verification_url': 'https://organic.ams.usda.gov',
                'requirements': ['95% organic ingredients', 'USDA approved certifier'],
                'valid_sectors': ['food', 'textiles', 'personal_care']
            },
            'Fair Trade Certified': {
                'issuer': 'Fair Trade USA',
                'verification_url': 'https://www.fairtradecertified.org',
                'requirements': ['Fair wages', 'Safe working conditions', 'Environmental protection'],
                'valid_sectors': ['food', 'textiles', 'home_goods']
            },
            'Energy Star': {
                'issuer': 'U.S. Environmental Protection Agency',
                'verification_url': 'https://www.energystar.gov',
                'requirements': ['Energy efficiency standards', 'EPA verification'],
                'valid_sectors': ['appliances', 'electronics', 'buildings']
            },
            'Forest Stewardship Council (FSC)': {
                'issuer': 'Forest Stewardship Council',
                'verification_url': 'https://fsc.org',
                'requirements': ['Sustainable forest management', 'Chain of custody'],
                'valid_sectors': ['paper', 'wood', 'textiles']
            }
        }
    
    def _load_certification_requirements(self) -> Dict[str, List[str]]:
        """Load requirements for each certification"""
        return {
            'USDA Organic': [
                'Must contain at least 95% organic ingredients',
                'Certified by USDA-accredited certifying agent',
                'No synthetic pesticides or fertilizers',
                'No GMOs'
            ],
            'Fair Trade Certified': [
                'Fair prices for producers',
                'Safe working conditions',
                'Environmental protection',
                'Community development'
            ],
            'Energy Star': [
                'Meet strict energy efficiency guidelines',
                'Verified by EPA',
                'Provide energy savings without sacrificing performance'
            ]
        }
    
    def validate_certifications(self, certifications: List[str]) -> Dict[str, Any]:
        """
        Validate claimed certifications
        
        Args:
            certifications: List of claimed certifications
            
        Returns:
            Validation results
        """
        validation = {
            'valid_certifications': [],
            'invalid_certifications': [],
            'unverified_certifications': [],
            'risk_score': 0.0,
            'verification_details': {}
        }
        
        for cert in certifications:
            if cert in self.valid_certifications:
                validation['valid_certifications'].append(cert)
                validation['verification_details'][cert] = {
                    'status': 'valid',
                    'issuer': self.valid_certifications[cert]['issuer'],
                    'verification_url': self.valid_certifications[cert]['verification_url']
                }
            else:
                # Check if it's a variation of a known certification
                if self._is_certification_variation(cert):
                    validation['unverified_certifications'].append(cert)
                    validation['verification_details'][cert] = {
                        'status': 'unverified',
                        'message': 'Certification name variation - needs verification'
                    }
                else:
                    validation['invalid_certifications'].append(cert)
                    validation['verification_details'][cert] = {
                        'status': 'invalid',
                        'message': 'Unknown or invalid certification'
                    }
                    validation['risk_score'] += 0.3
        
        # Calculate overall risk score
        total_certs = len(certifications)
        if total_certs > 0:
            invalid_ratio = len(validation['invalid_certifications']) / total_certs
            validation['risk_score'] = min(invalid_ratio, 1.0)
        
        return validation
    
    def _is_certification_variation(self, certification: str) -> bool:
        """Check if certification is a variation of a known one"""
        cert_lower = certification.lower()
        
        for valid_cert in self.valid_certifications.keys():
            valid_lower = valid_cert.lower()
            
            # Check for partial matches
            if any(word in cert_lower for word in valid_lower.split()):
                return True
        
        return False
    
    def get_certification_requirements(self, certification: str) -> List[str]:
        """Get requirements for a specific certification"""
        return self.certification_requirements.get(certification, [])
    
    def suggest_legitimate_certifications(self, product_category: str) -> List[str]:
        """Suggest legitimate certifications for a product category"""
        suggestions = []
        
        for cert, info in self.valid_certifications.items():
            if product_category.lower() in info.get('valid_sectors', []):
                suggestions.append(cert)
        
        return suggestions
