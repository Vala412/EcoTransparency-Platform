"""
Database models for the EcoTransparency Platform
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel

Base = declarative_base()

class Product(Base):
    """Product model for storing product information"""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    brand = Column(String(100), index=True)
    category = Column(String(100), index=True)
    price = Column(Float)
    
    # Sustainability metrics
    sustainability_score = Column(Float, index=True)
    carbon_footprint = Column(Float)  # kg CO2e
    water_usage = Column(Float)  # liters
    energy_consumption = Column(Float)  # kWh
    recyclability_score = Column(Float)
    
    # Supply chain data
    origin_country = Column(String(100))
    transportation_distance = Column(Float)  # km
    transportation_method = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sustainability_assessments = relationship("SustainabilityAssessment", back_populates="product")
    greenwashing_alerts = relationship("GreenwashingAlert", back_populates="product")
    demand_forecasts = relationship("DemandForecast", back_populates="product")

class SustainabilityAssessment(Base):
    """Sustainability assessment results"""
    __tablename__ = "sustainability_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    # Scores
    overall_score = Column(Float, nullable=False)
    emission_score = Column(Float)
    resource_usage_score = Column(Float)
    labor_practices_score = Column(Float)
    transportation_score = Column(Float)
    recyclability_score = Column(Float)
    
    # Explainability
    feature_importance = Column(JSON)
    recommendations = Column(JSON)
    
    # Metadata
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="sustainability_assessments")

class GreenwashingAlert(Base):
    """Greenwashing detection alerts"""
    __tablename__ = "greenwashing_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    # Alert details
    risk_score = Column(Float, nullable=False)  # 0-1 scale
    alert_type = Column(String(50))  # "text", "image", "claim"
    confidence = Column(Float)
    
    # Evidence
    flagged_text = Column(Text)
    flagged_claims = Column(JSON)
    missing_certifications = Column(JSON)
    
    # Actions
    status = Column(String(20), default="pending")  # pending, verified, dismissed
    action_taken = Column(String(100))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="greenwashing_alerts")

class DemandForecast(Base):
    """Demand forecasting results"""
    __tablename__ = "demand_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    # Forecast data
    forecast_date = Column(DateTime, nullable=False)
    predicted_demand = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    
    # Model info
    model_type = Column(String(50))
    model_version = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="demand_forecasts")

class User(Base):
    """User model for personalization"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    name = Column(String(100))
    
    # Preferences
    sustainability_priorities = Column(JSON)  # ["low_carbon", "plastic_free", "local"]
    price_sensitivity = Column(Float)  # 0-1 scale
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    interactions = relationship("UserInteraction", back_populates="user")
    recommendations = relationship("UserRecommendation", back_populates="user")

class UserInteraction(Base):
    """User interaction tracking"""
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    # Interaction details
    interaction_type = Column(String(50))  # "view", "purchase", "like", "dislike"
    rating = Column(Float)  # 1-5 scale
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="interactions")

class UserRecommendation(Base):
    """User recommendations"""
    __tablename__ = "user_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    # Recommendation details
    score = Column(Float, nullable=False)
    reason = Column(Text)
    recommendation_type = Column(String(50))  # "collaborative", "content", "hybrid"
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="recommendations")

# Pydantic models for API
class ProductBase(BaseModel):
    name: str
    description: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None

class ProductCreate(ProductBase):
    pass

class ProductResponse(ProductBase):
    id: int
    sustainability_score: Optional[float] = None
    carbon_footprint: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class SustainabilityAssessmentResponse(BaseModel):
    id: int
    product_id: int
    overall_score: float
    emission_score: Optional[float] = None
    resource_usage_score: Optional[float] = None
    labor_practices_score: Optional[float] = None
    transportation_score: Optional[float] = None
    recyclability_score: Optional[float] = None
    feature_importance: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class GreenwashingAlertResponse(BaseModel):
    id: int
    product_id: int
    risk_score: float
    alert_type: str
    confidence: Optional[float] = None
    flagged_text: Optional[str] = None
    flagged_claims: Optional[List[str]] = None
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class DemandForecastResponse(BaseModel):
    id: int
    product_id: int
    forecast_date: datetime
    predicted_demand: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    model_type: Optional[str] = None
    
    class Config:
        from_attributes = True

class GreenwashingAnalysisRequest(BaseModel):
    product_id: int
    description: Optional[str] = None
    marketing_text: Optional[str] = None
    image_paths: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": 1,
                "description": "Eco-friendly organic cotton t-shirt made from 100% sustainable materials",
                "marketing_text": "Save the planet with our green revolution clothing line",
                "image_paths": ["/path/to/image1.jpg"],
                "certifications": ["GOTS", "OEKO-TEX"]
            }
        }
