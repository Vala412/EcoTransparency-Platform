"""
Comprehensive test suite for EcoTransparency Platform
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestSustainabilityIndex:
    """Test sustainability index generation"""
    
    def test_sustainability_scoring(self):
        """Test sustainability scoring functionality"""
        from src.ml.sustainability_index import SustainabilityIndexGenerator
        
        # Create sample data
        data = {
            'carbon_footprint': [10.5, 25.3, 8.9],
            'water_usage': [150, 300, 100],
            'energy_consumption': [45, 80, 30],
            'transportation_distance': [200, 800, 150],
            'weight': [1.5, 3.0, 0.8],
            'packaging_recyclable': [1, 0, 1],
            'packaging_biodegradable': [0, 0, 1],
            'renewable_energy_percentage': [30, 10, 60]
        }
        
        df = pd.DataFrame(data)
        generator = SustainabilityIndexGenerator()
        
        # Test model training
        results = generator.train_model(df)
        assert 'r2_score' in results
        assert 'feature_importance' in results
        
        # Test prediction
        predictions = generator.predict_sustainability_score(df)
        assert 'sustainability_scores' in predictions
        assert 'explanations' in predictions
        assert len(predictions['sustainability_scores']) == len(df)
        
        print("✅ Sustainability scoring test passed")

class TestGreenwashingDetection:
    """Test greenwashing detection"""
    
    def test_greenwashing_analysis(self):
        """Test greenwashing detection functionality"""
        from src.ml.greenwashing_detection import GreenwashingDetector
        
        detector = GreenwashingDetector()
        
        # Test case with potential greenwashing
        test_data = {
            'description': 'Our product is 100% natural and completely eco-friendly with zero environmental impact.',
            'marketing_text': 'Studies show our product is the greenest on the market!',
            'certifications': ['USDA Organic', 'Fair Trade Certified']
        }
        
        result = detector.analyze_product_claims(test_data)
        
        assert 'overall_risk_score' in result
        assert 'risk_level' in result
        assert 'alerts' in result
        assert 'recommendations' in result
        assert isinstance(result['overall_risk_score'], float)
        assert result['risk_level'] in ['low', 'medium', 'high']
        
        print("✅ Greenwashing detection test passed")

class TestDemandForecasting:
    """Test demand forecasting"""
    
    def test_demand_forecasting(self):
        """Test demand forecasting functionality"""
        from src.ml.demand_forecasting import DemandForecastingEngine
        
        # Create sample demand data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        demand = 100 + np.random.normal(0, 10, 100)
        demand = np.maximum(demand, 0)
        
        demand_data = pd.DataFrame({
            'date': dates,
            'demand': demand,
            'product_id': ['product_1'] * 100
        })
        
        forecaster = DemandForecastingEngine()
        
        # Test data preparation
        prepared_data = forecaster.prepare_data(demand_data, 'date', 'demand', 'product_id')
        assert not prepared_data.empty
        
        # Test forecasting
        forecast = forecaster.forecast_demand('product_1', forecast_horizon=7)
        assert 'forecast' in forecast
        assert 'production_recommendations' in forecast
        assert len(forecast['forecast']) == 7
        
        print("✅ Demand forecasting test passed")

class TestPersonalization:
    """Test personalization engine"""
    
    def test_personalization_recommendations(self):
        """Test personalization recommendations"""
        from src.ml.personalization import PersonalizationEngine
        
        # Create sample interaction data
        interaction_data = pd.DataFrame({
            'user_id': ['user_1'] * 10,
            'product_id': range(1, 11),
            'interaction_type': ['view'] * 5 + ['purchase'] * 5,
            'rating': np.random.uniform(3, 5, 10),
            'category': ['Electronics'] * 5 + ['Home'] * 5,
            'sustainability_score': np.random.uniform(50, 90, 10),
            'price': np.random.uniform(20, 200, 10),
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D')
        })
        
        # Create sample products
        products_data = pd.DataFrame({
            'id': range(1, 21),
            'name': [f'Product {i}' for i in range(1, 21)],
            'category': ['Electronics'] * 10 + ['Home'] * 10,
            'price': np.random.uniform(50, 300, 20),
            'sustainability_score': np.random.uniform(40, 90, 20),
            'carbon_footprint': np.random.uniform(5, 25, 20)
        })
        
        engine = PersonalizationEngine()
        
        # Test user profile creation
        user_profile = engine.create_user_profile(
            'user_1',
            interaction_data,
            preferences={
                'sustainability_priorities': ['low_carbon', 'local'],
                'min_sustainability_score': 70
            }
        )
        
        assert 'interaction_count' in user_profile
        assert 'preferred_categories' in user_profile
        
        # Test recommendations
        recommendations = engine.generate_recommendations(
            'user_1',
            user_profile,
            products_data,
            num_recommendations=5
        )
        
        assert len(recommendations) <= 5
        assert all('product_id' in rec for rec in recommendations)
        assert all('score' in rec for rec in recommendations)
        
        print("✅ Personalization test passed")

class TestCarbonCalculator:
    """Test carbon footprint calculator"""
    
    def test_carbon_calculation(self):
        """Test carbon footprint calculation"""
        from src.utils.carbon_calculator import CarbonFootprintCalculator
        
        calculator = CarbonFootprintCalculator()
        
        # Test product footprint
        product_data = {
            'materials': {'plastic': 0.5, 'steel': 0.3, 'aluminum': 0.1},
            'manufacturing_energy': 25,
            'energy_mix': {'electricity': 0.7, 'natural_gas': 0.3},
            'packaging_materials': {'cardboard': 0.1, 'plastic': 0.05},
            'transportation': [
                {'distance': 500, 'weight': 1.0, 'mode': 'truck'},
                {'distance': 200, 'weight': 1.0, 'mode': 'train'}
            ],
            'weight': 1.0,
            'recycling_rate': 0.6,
            'use_energy_kwh': 0.1,
            'use_duration_years': 5
        }
        
        result = calculator.calculate_product_footprint(product_data)
        
        assert 'total_footprint_kg_co2e' in result
        assert 'per_unit_footprint' in result
        assert 'components' in result
        assert 'breakdown_percentage' in result
        assert result['total_footprint_kg_co2e'] > 0
        
        # Test basket footprint
        basket_products = [
            {**product_data, 'id': 1, 'name': 'Product A', 'quantity': 2},
            {**product_data, 'id': 2, 'name': 'Product B', 'quantity': 1}
        ]
        
        basket_result = calculator.calculate_basket_footprint(basket_products)
        
        assert 'total_footprint_kg_co2e' in basket_result
        assert 'comparisons' in basket_result
        assert 'recommendations' in basket_result
        assert basket_result['total_footprint_kg_co2e'] > 0
        
        print("✅ Carbon calculator test passed")

class TestDataPreprocessing:
    """Test data preprocessing"""
    
    def test_data_ingestion(self):
        """Test data ingestion pipeline"""
        from src.data.preprocessing import DataIngestionPipeline
        
        pipeline = DataIngestionPipeline()
        
        # Test CSV processing
        csv_data = pd.DataFrame({
            'product_id': [1, 2, 3],
            'name': ['Product A', 'Product B', 'Product C'],
            'category': ['Electronics', 'Home', 'Electronics'],
            'price': [100, 200, 150],
            'carbon_footprint': [10, 20, 15]
        })
        
        processed = pipeline.process_csv_data(csv_data)
        assert not processed.empty
        assert 'processed_date' in processed.columns
        
        # Test text processing
        text_data = "This is a sample product description with environmental claims."
        processed_text = pipeline.process_text_data(text_data)
        
        assert 'processed_text' in processed_text
        assert 'features' in processed_text
        
        print("✅ Data preprocessing test passed")

class TestConfiguration:
    """Test configuration"""
    
    def test_settings_loading(self):
        """Test settings loading"""
        from src.config.settings import Settings
        
        settings = Settings()
        
        # Test that settings object is created
        assert hasattr(settings, 'database_url')
        assert hasattr(settings, 'secret_key')
        assert hasattr(settings, 'algorithm')
        
        print("✅ Configuration test passed")

def run_all_tests():
    """Run all tests"""
    print("🧪 Running EcoTransparency Platform Tests")
    print("=" * 50)
    
    test_classes = [
        TestSustainabilityIndex,
        TestGreenwashingDetection,
        TestDemandForecasting,
        TestPersonalization,
        TestCarbonCalculator,
        TestDataPreprocessing,
        TestConfiguration
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        test_instance = test_class()
        methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method in methods:
            try:
                print(f"\n🔍 Running {test_class.__name__}.{method}")
                getattr(test_instance, method)()
                passed += 1
            except Exception as e:
                print(f"❌ {test_class.__name__}.{method} failed: {e}")
                failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} tests failed. Check the implementation.")

if __name__ == "__main__":
    run_all_tests()
