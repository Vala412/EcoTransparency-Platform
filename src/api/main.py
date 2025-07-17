"""
FastAPI main application for EcoTransparency Platform
"""

from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import json
import asyncio

# Import our modules
from ..config.settings import settings
from ..models.database import (
    ProductResponse, ProductCreate, SustainabilityAssessmentResponse,
    GreenwashingAlertResponse, GreenwashingAnalysisRequest, DemandForecastResponse
)
from ..ml.sustainability_index import SustainabilityIndexGenerator
from ..ml.greenwashing_detection import GreenwashingDetector
from ..ml.demand_forecasting import DemandForecastingEngine
from ..ml.personalization import PersonalizationEngine
from ..data.preprocessing import DataIngestionPipeline
from ..utils.logger import setup_logger

# Setup logging
logger = setup_logger()

# Global variables for ML models
sustainability_model = None
greenwashing_detector = None
demand_forecaster = None
personalization_engine = None
data_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global sustainability_model, greenwashing_detector, demand_forecaster, personalization_engine, data_pipeline
    
    logger.info("Starting EcoTransparency Platform...")
    
    # Initialize models
    sustainability_model = SustainabilityIndexGenerator()
    greenwashing_detector = GreenwashingDetector()
    demand_forecaster = DemandForecastingEngine()
    personalization_engine = PersonalizationEngine()
    data_pipeline = DataIngestionPipeline()
    
    # Load pre-trained models if available
    try:
        # In production, load from saved model files
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
    
    yield
    
    logger.info("Shutting down EcoTransparency Platform...")

# Create FastAPI app
app = FastAPI(
    title="EcoTransparency Platform API",
    description="API for sustainability scoring, greenwashing detection, and demand forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Product management endpoints
@app.post("/products/", response_model=ProductResponse)
async def create_product(product: ProductCreate):
    """Create a new product"""
    try:
        # In production, save to database
        product_data = product.dict()
        product_data['id'] = np.random.randint(1000, 9999)
        product_data['created_at'] = datetime.now()
        
        return ProductResponse(**product_data)
    except Exception as e:
        logger.error(f"Error creating product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int):
    """Get product by ID"""
    try:
        # In production, fetch from database
        # Mock product data
        product_data = {
            "id": product_id,
            "name": f"Product {product_id}",
            "description": "Sample product description",
            "brand": "EcoFriendly Brand",
            "category": "Electronics",
            "price": 99.99,
            "sustainability_score": 75.0,
            "carbon_footprint": 12.5,
            "created_at": datetime.now()
        }
        
        return ProductResponse(**product_data)
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {e}")
        raise HTTPException(status_code=404, detail="Product not found")

@app.get("/products/", response_model=List[ProductResponse])
async def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = None,
    min_sustainability_score: Optional[float] = None
):
    """List products with filtering"""
    try:
        # In production, fetch from database with filters
        # Mock product data
        products = []
        for i in range(skip, min(skip + limit, skip + 20)):
            product_data = {
                "id": i + 1,
                "name": f"Product {i + 1}",
                "description": f"Sample product {i + 1} description",
                "brand": "EcoFriendly Brand",
                "category": "Electronics" if i % 2 == 0 else "Home",
                "price": 50.0 + (i * 10),
                "sustainability_score": 50.0 + (i * 5) % 50,
                "carbon_footprint": 5.0 + (i * 2) % 20,
                "created_at": datetime.now()
            }
            
            # Apply filters
            if category and product_data["category"] != category:
                continue
            if min_sustainability_score and product_data["sustainability_score"] < min_sustainability_score:
                continue
            
            products.append(ProductResponse(**product_data))
        
        return products
    except Exception as e:
        logger.error(f"Error listing products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sustainability scoring endpoints
@app.post("/sustainability/score", response_model=SustainabilityAssessmentResponse)
async def score_product_sustainability(product_id: int, product_data: Dict[str, Any]):
    """Score product sustainability"""
    try:
        # Create DataFrame from product data
        df = pd.DataFrame([product_data])
        
        # Generate sustainability score
        result = sustainability_model.predict_sustainability_score(df)
        
        assessment = {
            "id": np.random.randint(1000, 9999),
            "product_id": product_id,
            "overall_score": result['sustainability_scores'][0],
            "emission_score": result['component_scores'][0]['emissions'],
            "resource_usage_score": result['component_scores'][0]['resource_usage'],
            "transportation_score": result['component_scores'][0]['transportation'],
            "recyclability_score": result['component_scores'][0]['recyclability'],
            "labor_practices_score": result['component_scores'][0]['labor_practices'],
            "feature_importance": result['explanations'][0]['top_influencing_factors'],
            "recommendations": result['explanations'][0]['recommendations'],
            "created_at": datetime.now()
        }
        
        return SustainabilityAssessmentResponse(**assessment)
    except Exception as e:
        logger.error(f"Error scoring product sustainability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sustainability/benchmark/{product_id}")
async def get_sustainability_benchmark(product_id: int, industry: Optional[str] = None):
    """Get sustainability benchmark for product"""
    try:
        # Mock benchmark data
        benchmark = {
            "product_id": product_id,
            "industry_average": 65.0,
            "category_average": 70.0,
            "percentile_rank": 75.0,
            "top_performers": [
                {"name": "Top Product 1", "score": 95.0},
                {"name": "Top Product 2", "score": 92.0},
                {"name": "Top Product 3", "score": 88.0}
            ],
            "improvement_areas": [
                "Reduce carbon footprint",
                "Improve packaging recyclability",
                "Source materials locally"
            ]
        }
        
        return benchmark
    except Exception as e:
        logger.error(f"Error getting benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Greenwashing detection endpoints
@app.post("/greenwashing/analyze", response_model=GreenwashingAlertResponse)
async def analyze_greenwashing(request: GreenwashingAnalysisRequest):
    """Analyze product for greenwashing"""
    try:
        # Prepare analysis data
        analysis_data = {
            'id': request.product_id,
            'description': request.description or '',
            'marketing_text': request.marketing_text or '',
            'image_paths': request.image_paths or [],
            'certifications': request.certifications or []
        }
        
        # Analyze product claims
        result = greenwashing_detector.analyze_product_claims(analysis_data)
        
        if result['overall_risk_score'] > 0.3:  # Only create alert if risk is significant
            alert = {
                "id": np.random.randint(1000, 9999),
                "product_id": request.product_id,
                "risk_score": result['overall_risk_score'],
                "alert_type": "text_analysis",
                "confidence": 0.85,
                "flagged_text": str(result.get('text_analysis', {}).get('flagged_phrases', [])),
                "flagged_claims": result.get('text_analysis', {}).get('red_flags', []),
                "status": "pending",
                "created_at": datetime.now()
            }
            
            return GreenwashingAlertResponse(**alert)
        else:
            # No significant risk found
            return JSONResponse(
                content={"message": "No significant greenwashing risk detected"},
                status_code=200
            )
    except Exception as e:
        logger.error(f"Error analyzing greenwashing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/greenwashing/alerts/{product_id}")
async def get_greenwashing_alerts(product_id: int):
    """Get greenwashing alerts for a product"""
    try:
        # In production, fetch from database
        # Mock alert data
        alerts = [
            {
                "id": 1,
                "product_id": product_id,
                "risk_score": 0.65,
                "alert_type": "text_analysis",
                "confidence": 0.85,
                "flagged_text": "100% natural ingredients",
                "flagged_claims": ["absolute_claims", "vague_claims"],
                "status": "pending",
                "created_at": datetime.now()
            }
        ]
        
        return [GreenwashingAlertResponse(**alert) for alert in alerts]
    except Exception as e:
        logger.error(f"Error getting greenwashing alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Demand forecasting endpoints
@app.post("/forecasting/train")
async def train_demand_model(product_id: int, training_data: Dict[str, Any]):
    """Train demand forecasting model"""
    try:
        # Convert training data to DataFrame
        df = pd.DataFrame(training_data['historical_data'])
        
        # Train model
        results = demand_forecaster.train_multiple_models(
            df, 
            date_col='date',
            demand_col='demand',
            product_id=str(product_id)
        )
        
        return {
            "message": "Model trained successfully",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error training demand model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecasting/predict/{product_id}", response_model=List[DemandForecastResponse])
async def predict_demand(product_id: int, forecast_horizon: int = Query(30, ge=1, le=365)):
    """Predict demand for a product"""
    try:
        # Generate forecast
        forecast_result = demand_forecaster.forecast_demand(
            product_id=str(product_id),
            forecast_horizon=forecast_horizon
        )
        
        # Convert to response format
        forecasts = []
        for i, (date, demand) in enumerate(zip(forecast_result['dates'], forecast_result['forecast'])):
            forecast = {
                "id": i + 1,
                "product_id": product_id,
                "forecast_date": date,
                "predicted_demand": demand,
                "confidence_interval_lower": forecast_result.get('lower_bound', [None])[i] if forecast_result.get('lower_bound') else None,
                "confidence_interval_upper": forecast_result.get('upper_bound', [None])[i] if forecast_result.get('upper_bound') else None,
                "model_type": forecast_result['model_type']
            }
            forecasts.append(DemandForecastResponse(**forecast))
        
        return forecasts
    except Exception as e:
        logger.error(f"Error predicting demand: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecasting/production-recommendations/{product_id}")
async def get_production_recommendations(product_id: int, forecast_horizon: int = Query(30, ge=1, le=365)):
    """Get production recommendations based on demand forecast"""
    try:
        # Get forecast
        forecast_result = demand_forecaster.forecast_demand(
            product_id=str(product_id),
            forecast_horizon=forecast_horizon
        )
        
        return forecast_result.get('production_recommendations', {})
    except Exception as e:
        logger.error(f"Error getting production recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Personalization endpoints
@app.post("/personalization/profile/{user_id}")
async def create_user_profile(user_id: str, profile_data: Dict[str, Any]):
    """Create user profile"""
    try:
        # Convert interaction data to DataFrame
        interaction_df = pd.DataFrame(profile_data.get('interaction_data', []))
        preferences = profile_data.get('preferences', {})
        
        # Create profile
        profile = personalization_engine.create_user_profile(
            user_id=user_id,
            interaction_data=interaction_df,
            preferences=preferences
        )
        
        return profile
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/personalization/recommendations/{user_id}")
async def get_personalized_recommendations(
    user_id: str,
    num_recommendations: int = Query(10, ge=1, le=50),
    category: Optional[str] = None
):
    """Get personalized recommendations for user"""
    try:
        # In production, fetch user profile from database
        # Mock user profile
        user_profile = {
            "user_id": user_id,
            "preferences": {"sustainability_priorities": ["low_carbon", "local"]},
            "purchase_history": [],
            "preferred_categories": ["Electronics", "Home"],
            "min_sustainability_score": 60,
            "sustainability_profile": {
                "priorities": ["low_carbon", "local"],
                "min_score_threshold": 60,
                "preferred_certifications": ["ENERGY_STAR", "FAIR_TRADE"],
                "eco_friendly_tendency": 0.8
            }
        }
        
        # Mock products DataFrame
        products_data = []
        for i in range(1, 101):
            products_data.append({
                "id": i,
                "name": f"Product {i}",
                "category": "Electronics" if i % 2 == 0 else "Home",
                "price": 50.0 + (i * 10),
                "sustainability_score": 50.0 + (i * 5) % 50,
                "carbon_footprint": 5.0 + (i * 2) % 20
            })
        
        products_df = pd.DataFrame(products_data)
        
        # Generate recommendations
        recommendations = personalization_engine.generate_recommendations(
            user_id=user_id,
            user_profile=user_profile,
            products_df=products_df,
            num_recommendations=num_recommendations
        )
        
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/personalization/feedback/{user_id}")
async def update_user_feedback(user_id: str, feedback_data: Dict[str, Any]):
    """Update user feedback"""
    try:
        personalization_engine.update_user_feedback(
            user_id=user_id,
            product_id=feedback_data['product_id'],
            feedback_type=feedback_data['feedback_type'],
            rating=feedback_data.get('rating'),
            implicit_feedback=feedback_data.get('implicit_feedback', False)
        )
        
        return {"message": "Feedback updated successfully"}
    except Exception as e:
        logger.error(f"Error updating feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/personalization/trending")
async def get_trending_sustainable_products():
    """Get trending sustainable products"""
    try:
        # Mock products DataFrame
        products_data = []
        for i in range(1, 21):
            products_data.append({
                "id": i,
                "name": f"Sustainable Product {i}",
                "category": "Electronics" if i % 2 == 0 else "Home",
                "sustainability_score": 70 + (i * 2) % 30
            })
        
        products_df = pd.DataFrame(products_data)
        
        trending = personalization_engine.get_trending_sustainable_products(products_df)
        
        return trending
    except Exception as e:
        logger.error(f"Error getting trending products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Carbon footprint simulation endpoints
@app.post("/carbon/calculate")
async def calculate_carbon_footprint(calculation_data: Dict[str, Any]):
    """Calculate carbon footprint for products or basket"""
    try:
        products = calculation_data.get('products', [])
        total_footprint = 0
        
        for product in products:
            # Mock carbon footprint calculation
            weight = product.get('weight', 1)
            distance = product.get('transportation_distance', 100)
            material_factor = product.get('material_carbon_factor', 2.5)
            
            product_footprint = weight * distance * material_factor * 0.001  # Convert to kg CO2e
            total_footprint += product_footprint
        
        # Create comparison
        comparison = {
            "equivalent_car_miles": total_footprint * 2.5,
            "equivalent_ac_hours": total_footprint * 0.5,
            "trees_to_offset": total_footprint * 0.04
        }
        
        return {
            "total_footprint_kg_co2e": total_footprint,
            "per_product_footprint": [total_footprint / len(products)] * len(products),
            "comparison": comparison,
            "recommendations": [
                "Choose products with lower carbon footprint",
                "Consider local alternatives",
                "Offset remaining emissions"
            ]
        }
    except Exception as e:
        logger.error(f"Error calculating carbon footprint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data upload endpoints
@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...), data_type: str = Query(...)):
    """Upload data for processing"""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Process based on data type
        if data_type == "product_data":
            # Process product data
            df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
            processed_df = data_pipeline.ingest_product_data(df)
            
            return {
                "message": "Product data uploaded successfully",
                "records_processed": len(processed_df),
                "columns": list(processed_df.columns)
            }
        elif data_type == "demand_data":
            # Process demand data
            df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
            processed_df = data_pipeline.format_time_series_data(df, 'date', 'demand')
            
            return {
                "message": "Demand data uploaded successfully",
                "records_processed": len(processed_df),
                "date_range": f"{processed_df.index.min()} to {processed_df.index.max()}"
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid data type")
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/dashboard")
async def get_dashboard_analytics():
    """Get dashboard analytics"""
    try:
        # Mock analytics data
        analytics = {
            "total_products": 1234,
            "avg_sustainability_score": 67.5,
            "greenwashing_alerts": 89,
            "carbon_footprint_reduction": 15.2,
            "recent_activity": [
                {"type": "product_added", "count": 45, "date": "2024-01-15"},
                {"type": "sustainability_scored", "count": 123, "date": "2024-01-15"},
                {"type": "greenwashing_detected", "count": 12, "date": "2024-01-15"}
            ],
            "top_categories": [
                {"category": "Electronics", "avg_score": 72.1, "count": 345},
                {"category": "Home", "avg_score": 68.3, "count": 289},
                {"category": "Clothing", "avg_score": 64.7, "count": 234}
            ]
        }
        
        return analytics
    except Exception as e:
        logger.error(f"Error getting dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
