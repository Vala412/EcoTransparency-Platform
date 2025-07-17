"""
Example usage scripts for the EcoTransparency Platform
"""

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import json

# Example 1: Basic sustainability scoring
def example_sustainability_scoring():
    """Example of sustainability scoring"""
    print("=== Sustainability Scoring Example ===")
    
    # Import the sustainability index generator
    from src.ml.sustainability_index import SustainabilityIndexGenerator
    
    # Create sample product data
    products_data = {
        'carbon_footprint': [10.5, 25.3, 8.9, 30.1, 12.0],
        'water_usage': [150, 300, 100, 400, 180],
        'energy_consumption': [45, 80, 30, 95, 50],
        'transportation_distance': [200, 800, 150, 1200, 300],
        'weight': [1.5, 3.0, 0.8, 4.5, 2.0],
        'packaging_recyclable': [1, 0, 1, 0, 1],
        'packaging_biodegradable': [0, 0, 1, 0, 1],
        'renewable_energy_percentage': [30, 10, 60, 5, 45]
    }
    
    df = pd.DataFrame(products_data)
    
    # Initialize the sustainability index generator
    generator = SustainabilityIndexGenerator()
    
    # Train the model (in practice, you'd use a larger dataset)
    print("Training sustainability model...")
    training_results = generator.train_model(df)
    print(f"Model trained with R2 score: {training_results['r2_score']:.3f}")
    
    # Predict sustainability scores
    print("\nGenerating sustainability scores...")
    results = generator.predict_sustainability_score(df)
    
    # Display results
    for i, score in enumerate(results['sustainability_scores']):
        print(f"Product {i+1}: Sustainability Score = {score:.1f}")
        print(f"  Grade: {results['explanations'][i]['grade']}")
        print(f"  Strengths: {', '.join(results['explanations'][i]['strengths'])}")
        print(f"  Recommendations: {', '.join(results['explanations'][i]['recommendations'][:2])}")
        print()

# Example 2: Greenwashing detection
def example_greenwashing_detection():
    """Example of greenwashing detection"""
    print("=== Greenwashing Detection Example ===")
    
    from src.ml.greenwashing_detection import GreenwashingDetector
    
    # Initialize detector
    detector = GreenwashingDetector()
    
    # Test cases
    test_cases = [
        {
            'id': 1,
            'description': 'Our product is 100% natural and completely eco-friendly with zero environmental impact.',
            'marketing_text': 'Studies show our product is the greenest on the market!',
            'certifications': ['USDA Organic', 'Fair Trade Certified']
        },
        {
            'id': 2,
            'description': 'Made with 95% recycled materials, carbon-neutral shipping, and biodegradable packaging.',
            'marketing_text': 'Certified by Forest Stewardship Council (FSC) and Energy Star rated.',
            'certifications': ['FSC Certified', 'Energy Star']
        },
        {
            'id': 3,
            'description': 'Chemical-free, non-toxic, and environmentally safe product.',
            'marketing_text': 'Experts recommend our green solution for conscious consumers.',
            'certifications': ['Green Seal', 'EcoLabel']
        }
    ]
    
    for test_case in test_cases:
        print(f"\nAnalyzing Product {test_case['id']}:")
        print(f"Description: {test_case['description'][:60]}...")
        
        result = detector.analyze_product_claims(test_case)
        
        print(f"Risk Score: {result['overall_risk_score']:.2f}")
        print(f"Risk Level: {result['risk_level'].upper()}")
        
        if result['alerts']:
            print("Alerts:")
            for alert in result['alerts']:
                print(f"  - {alert['type']}: {alert['message']}")
        
        if result['recommendations']:
            print("Recommendations:")
            for rec in result['recommendations'][:2]:
                print(f"  - {rec}")

# Example 3: Demand forecasting
def example_demand_forecasting():
    """Example of demand forecasting"""
    print("=== Demand Forecasting Example ===")
    
    from src.ml.demand_forecasting import DemandForecastingEngine
    
    # Create sample demand data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate synthetic demand with trend and seasonality
    np.random.seed(42)
    base_demand = 100
    trend = np.linspace(0, 20, len(dates))
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 5, len(dates))
    
    demand = base_demand + trend + seasonal + noise
    demand = np.maximum(demand, 0)  # Ensure non-negative
    
    demand_data = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'product_id': ['product_1'] * len(dates)
    })
    
    print(f"Generated {len(demand_data)} days of demand data")
    print(f"Average demand: {demand_data['demand'].mean():.1f}")
    print(f"Demand range: {demand_data['demand'].min():.1f} - {demand_data['demand'].max():.1f}")
    
    # Initialize forecasting engine
    forecaster = DemandForecastingEngine()
    
    # Prepare data
    prepared_data = forecaster.prepare_data(demand_data, 'date', 'demand', 'product_id')
    
    # Train models
    print("\nTraining demand forecasting models...")
    results = forecaster.train_multiple_models(
        prepared_data, 
        date_col='date', 
        demand_col='demand',
        product_id='product_1'
    )
    
    print(f"Models trained: {', '.join(results['models_trained'])}")
    print(f"Best model: {results['best_model']}")
    
    # Generate forecast
    print("\nGenerating 30-day forecast...")
    forecast = forecaster.forecast_demand('product_1', forecast_horizon=30)
    
    print(f"Total forecasted demand: {sum(forecast['forecast']):.1f}")
    print(f"Average daily demand: {np.mean(forecast['forecast']):.1f}")
    print(f"Peak demand: {max(forecast['forecast']):.1f}")
    
    # Show production recommendations
    recommendations = forecast['production_recommendations']
    print(f"\nProduction recommendations:")
    print(f"  Recommended production: {recommendations['recommended_production']:.1f}")
    print(f"  Safety stock: {recommendations['safety_stock']:.1f}")

# Example 4: Personalized recommendations
def example_personalization():
    """Example of personalized recommendations"""
    print("=== Personalization Example ===")
    
    from src.ml.personalization import PersonalizationEngine
    
    # Create sample user interaction data
    interaction_data = pd.DataFrame({
        'user_id': ['user_1'] * 20,
        'product_id': range(1, 21),
        'interaction_type': ['view'] * 10 + ['purchase'] * 5 + ['like'] * 5,
        'rating': np.random.uniform(3, 5, 20),
        'category': ['Electronics'] * 10 + ['Home'] * 10,
        'sustainability_score': np.random.uniform(50, 90, 20),
        'price': np.random.uniform(20, 200, 20),
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D')
    })
    
    # Create sample products
    products_data = []
    for i in range(1, 101):
        products_data.append({
            'id': i,
            'name': f'Product {i}',
            'category': 'Electronics' if i % 2 == 0 else 'Home',
            'price': 50 + (i * 5),
            'sustainability_score': 40 + (i % 50),
            'carbon_footprint': 5 + (i % 20)
        })
    
    products_df = pd.DataFrame(products_data)
    
    # Initialize personalization engine
    engine = PersonalizationEngine()
    
    # Create user profile
    print("Creating user profile...")
    user_profile = engine.create_user_profile(
        'user_1', 
        interaction_data,
        preferences={
            'sustainability_priorities': ['low_carbon', 'local'],
            'min_sustainability_score': 70
        }
    )
    
    print(f"User profile created:")
    print(f"  Interaction count: {user_profile['interaction_count']}")
    print(f"  Preferred categories: {user_profile['preferred_categories']}")
    print(f"  Price sensitivity: {user_profile['price_sensitivity']:.2f}")
    
    # Generate recommendations
    print("\nGenerating personalized recommendations...")
    recommendations = engine.generate_recommendations(
        'user_1',
        user_profile,
        products_df,
        num_recommendations=5
    )
    
    print(f"Top 5 recommendations:")
    for i, rec in enumerate(recommendations[:5]):
        print(f"  {i+1}. Product {rec['product_id']} (Score: {rec['score']:.2f})")
        print(f"     Explanation: {rec['explanation']}")

# Example 5: Carbon footprint calculation
def example_carbon_calculation():
    """Example of carbon footprint calculation"""
    print("=== Carbon Footprint Calculation Example ===")
    
    from src.utils.carbon_calculator import CarbonFootprintCalculator
    
    # Initialize calculator
    calculator = CarbonFootprintCalculator()
    
    # Example product data
    product_data = {
        'materials': {'plastic': 0.5, 'steel': 0.3, 'aluminum': 0.1},
        'manufacturing_energy': 25,  # kWh
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
    
    # Calculate footprint
    result = calculator.calculate_product_footprint(product_data)
    
    print(f"Total carbon footprint: {result['total_footprint_kg_co2e']:.2f} kg CO2e")
    print(f"Per unit footprint: {result['per_unit_footprint']:.2f} kg CO2e")
    
    print("\nFootprint breakdown:")
    for component, value in result['components'].items():
        percentage = result['breakdown_percentage'][component]
        print(f"  {component}: {value:.2f} kg CO2e ({percentage:.1f}%)")
    
    # Calculate basket footprint
    print("\n--- Basket Calculation ---")
    
    basket_products = [
        {**product_data, 'id': 1, 'name': 'Product A', 'quantity': 2},
        {**product_data, 'id': 2, 'name': 'Product B', 'quantity': 1, 'weight': 0.5},
        {**product_data, 'id': 3, 'name': 'Product C', 'quantity': 3, 'weight': 1.5}
    ]
    
    basket_result = calculator.calculate_basket_footprint(basket_products)
    
    print(f"Basket total footprint: {basket_result['total_footprint_kg_co2e']:.2f} kg CO2e")
    
    print("\nComparisons:")
    comparisons = basket_result['comparisons']
    print(f"  Equivalent to {comparisons['car_miles_equivalent']:.1f} car miles")
    print(f"  Equivalent to {comparisons['trees_to_offset']:.1f} trees to offset")
    
    print("\nRecommendations:")
    for rec in basket_result['recommendations'][:3]:
        print(f"  - {rec}")

# Example 6: Running the FastAPI server
def example_api_usage():
    """Example of API usage"""
    print("=== API Usage Example ===")
    print("To run the FastAPI server:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the server: python main.py")
    print("3. Access the API at: http://localhost:8000")
    print("4. View API documentation at: http://localhost:8000/docs")
    
    print("\nExample API calls:")
    print("- GET /health - Health check")
    print("- GET /products/ - List products")
    print("- POST /sustainability/score - Score product sustainability")
    print("- POST /greenwashing/analyze - Analyze for greenwashing")
    print("- GET /forecasting/predict/{product_id} - Get demand forecast")
    print("- GET /personalization/recommendations/{user_id} - Get recommendations")

# Example 7: Running the Streamlit dashboard
def example_streamlit_usage():
    """Example of Streamlit dashboard usage"""
    print("=== Streamlit Dashboard Example ===")
    print("To run the Streamlit dashboard:")
    print("1. Install dependencies: pip install streamlit plotly")
    print("2. Run the dashboard: streamlit run frontend/streamlit_app.py")
    print("3. Access the dashboard at: http://localhost:8501")
    
    print("\nDashboard features:")
    print("- 📊 Dashboard - Platform overview and analytics")
    print("- 🔍 Product Analysis - Detailed product information")
    print("- 🌿 Sustainability Scoring - Calculate sustainability scores")
    print("- 🕵️ Greenwashing Detection - Analyze product claims")
    print("- 📈 Demand Forecasting - Predict future demand")
    print("- 👤 Personalized Recommendations - Get tailored suggestions")
    print("- 🌍 Carbon Calculator - Calculate carbon footprint")
    print("- 📤 Data Upload - Upload and process data")

def main():
    """Run all examples"""
    print("EcoTransparency Platform - Example Usage")
    print("=" * 50)
    
    examples = [
        example_sustainability_scoring,
        example_greenwashing_detection,
        example_demand_forecasting,
        example_personalization,
        example_carbon_calculation,
        example_api_usage,
        example_streamlit_usage
    ]
    
    for i, example in enumerate(examples):
        print(f"\n{i+1}. ", end="")
        try:
            example()
        except Exception as e:
            print(f"Error running example: {e}")
            print("Note: Some examples require dependencies to be installed")
        
        if i < len(examples) - 1:
            print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
