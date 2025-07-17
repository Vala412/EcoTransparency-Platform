# EcoTransparency Platform

A comprehensive sustainability platform for environmental transparency, waste reduction, and greenwashing detection.

## Features

### 🌱 Core Modules
- **Data Ingestion**: Multi-source data processing (CSV, APIs, images, text)
- **Sustainability Scoring**: ML-powered sustainability index generation
- **Greenwashing Detection**: Advanced NLP and computer vision analysis
- **Demand Forecasting**: Time series prediction using multiple models
- **Personalization**: Recommendation engine for sustainable products
- **Carbon Footprint Calculator**: Detailed lifecycle carbon assessment
- **Interactive Dashboard**: Streamlit-based web interface
- **RESTful API**: FastAPI-based backend with comprehensive endpoints

### 🔧 Technical Stack
- **Backend**: FastAPI, Python 3.8+
- **Frontend**: Streamlit
- **Database**: PostgreSQL, MongoDB
- **ML/AI**: scikit-learn, XGBoost, Prophet, PyTorch
- **Computer Vision**: OpenCV, Tesseract OCR, PIL
- **NLP**: spaCy, Transformers, NLTK
- **Time Series**: Prophet, ARIMA, LSTM
- **Visualization**: Plotly, Matplotlib, Seaborn

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Project

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Database Setup

```bash
# PostgreSQL setup (using Docker)
docker run -d --name postgres \
  -e POSTGRES_DB=ecotransparency \
  -e POSTGRES_USER=eco_user \
  -e POSTGRES_PASSWORD=eco_password \
  -p 5432:5432 postgres:13

# MongoDB setup (using Docker)
docker run -d --name mongodb \
  -p 27017:27017 mongo:4.4
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://eco_user:eco_password@localhost/ecotransparency
MONGODB_URL=mongodb://localhost:27017/ecotransparency

# API
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# External APIs
WEATHER_API_KEY=your-weather-api-key
SUPPLY_CHAIN_API_KEY=your-supply-chain-api-key
```

### 4. Run the Application

#### Option A: FastAPI Backend
```bash
# Run the FastAPI server
python main.py

# Access API documentation
# http://localhost:8000/docs
```

#### Option B: Streamlit Dashboard
```bash
# Run the Streamlit dashboard
streamlit run frontend/streamlit_app.py

# Access dashboard
# http://localhost:8501
```

## Project Structure

```
Project/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables
├── README.md                 # Project documentation
├── examples/
│   └── usage_examples.py     # Example usage scripts
├── src/
│   ├── config/
│   │   └── settings.py       # Configuration management
│   ├── models/
│   │   └── database.py       # Database models
│   ├── data/
│   │   └── preprocessing.py  # Data ingestion & preprocessing
│   ├── ml/
│   │   ├── sustainability_index.py    # Sustainability scoring
│   │   ├── greenwashing_detection.py  # Greenwashing detection
│   │   ├── demand_forecasting.py      # Demand forecasting
│   │   └── personalization.py         # Recommendation engine
│   ├── api/
│   │   └── main.py          # FastAPI application
│   └── utils/
│       ├── logger.py        # Logging utilities
│       └── carbon_calculator.py  # Carbon footprint calculation
└── frontend/
    └── streamlit_app.py     # Streamlit dashboard
```

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /products/` - List all products
- `POST /products/` - Create new product
- `GET /products/{product_id}` - Get product details

### Sustainability Scoring
- `POST /sustainability/score` - Calculate sustainability score
- `GET /sustainability/score/{product_id}` - Get product sustainability score

### Greenwashing Detection
- `POST /greenwashing/analyze` - Analyze product claims
- `GET /greenwashing/alerts` - Get greenwashing alerts

### Demand Forecasting
- `GET /forecasting/predict/{product_id}` - Get demand forecast
- `POST /forecasting/train` - Train forecasting model

### Personalization
- `GET /personalization/recommendations/{user_id}` - Get personalized recommendations
- `POST /personalization/profile` - Create/update user profile

### Carbon Footprint
- `POST /carbon/calculate` - Calculate product carbon footprint
- `POST /carbon/basket` - Calculate basket carbon footprint

## Usage Examples

### 1. Sustainability Scoring
```python
from src.ml.sustainability_index import SustainabilityIndexGenerator

# Initialize generator
generator = SustainabilityIndexGenerator()

# Prepare product data
product_data = {
    'carbon_footprint': 10.5,
    'water_usage': 150,
    'energy_consumption': 45,
    'transportation_distance': 200,
    'packaging_recyclable': 1,
    'renewable_energy_percentage': 30
}

# Generate score
result = generator.predict_sustainability_score(product_data)
print(f"Sustainability Score: {result['sustainability_score']:.1f}")
```

### 2. Greenwashing Detection
```python
from src.ml.greenwashing_detection import GreenwashingDetector

# Initialize detector
detector = GreenwashingDetector()

# Analyze product claims
result = detector.analyze_product_claims({
    'description': 'Our product is 100% natural and eco-friendly',
    'marketing_text': 'Studies show our product is the greenest!',
    'certifications': ['USDA Organic']
})

print(f"Risk Level: {result['risk_level']}")
print(f"Alerts: {result['alerts']}")
```

### 3. Demand Forecasting
```python
from src.ml.demand_forecasting import DemandForecastingEngine

# Initialize forecaster
forecaster = DemandForecastingEngine()

# Generate 30-day forecast
forecast = forecaster.forecast_demand('product_1', forecast_horizon=30)
print(f"Average daily demand: {np.mean(forecast['forecast']):.1f}")
```

### 4. Carbon Footprint Calculation
```python
from src.utils.carbon_calculator import CarbonFootprintCalculator

# Initialize calculator
calculator = CarbonFootprintCalculator()

# Calculate product footprint
result = calculator.calculate_product_footprint({
    'materials': {'plastic': 0.5, 'steel': 0.3},
    'manufacturing_energy': 25,
    'transportation': [{'distance': 500, 'weight': 1.0, 'mode': 'truck'}],
    'weight': 1.0
})

print(f"Total footprint: {result['total_footprint_kg_co2e']:.2f} kg CO2e")
```

## Dashboard Features

### 📊 Main Dashboard
- Platform overview and key metrics
- Real-time analytics and trends
- Interactive charts and visualizations

### 🔍 Product Analysis
- Detailed product information
- Sustainability scoring and explanations
- Greenwashing risk assessment

### 📈 Demand Forecasting
- Interactive demand prediction
- Multiple forecasting models
- Production recommendations

### 👤 Personalization
- User profile management
- Personalized recommendations
- Sustainability matching

### 🌍 Carbon Calculator
- Product carbon footprint calculation
- Basket-level carbon assessment
- Offsetting recommendations

## Configuration

### Database Configuration
```python
# PostgreSQL (primary database)
DATABASE_URL = "postgresql://user:password@localhost/ecotransparency"

# MongoDB (for document storage)
MONGODB_URL = "mongodb://localhost:27017/ecotransparency"
```

### ML Model Configuration
```python
# Sustainability scoring
SUSTAINABILITY_MODEL_PATH = "models/sustainability_model.pkl"

# Greenwashing detection
GREENWASHING_MODEL_PATH = "models/greenwashing_model.pkl"

# Demand forecasting
FORECASTING_MODEL_PATH = "models/forecasting_model.pkl"
```

## Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_sustainability.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Docker Deployment
```bash
# Build image
docker build -t ecotransparency-platform .

# Run container
docker run -p 8000:8000 ecotransparency-platform
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact:
- Email: support@ecotransparency.com
- Documentation: [docs.ecotransparency.com](https://docs.ecotransparency.com)
- Issues: [GitHub Issues](https://github.com/your-org/ecotransparency/issues)

## Acknowledgments

- OpenAI for AI/ML capabilities
- Streamlit for the dashboard framework
- FastAPI for the backend framework
- The open-source community for various libraries and tools

1. Start the FastAPI server: `uvicorn api.main:app --reload`
2. Access the dashboard at `http://localhost:8000`
3. Upload product data or use the demo dataset
4. View sustainability scores, greenwashing alerts, and personalized recommendations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
