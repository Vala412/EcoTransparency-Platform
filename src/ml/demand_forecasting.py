"""
Demand Forecasting Module using Time Series Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class DemandForecastingEngine:
    """Main demand forecasting engine"""
    
    def __init__(self):
        self.models = {}
        self.model_performances = {}
        self.feature_engineering = FeatureEngineer()
        self.external_data = ExternalDataCollector()
        
    def prepare_data(self, df: pd.DataFrame, date_col: str, demand_col: str, 
                    product_id_col: str = None) -> pd.DataFrame:
        """
        Prepare data for demand forecasting
        
        Args:
            df: DataFrame with historical demand data
            date_col: Name of the date column
            demand_col: Name of the demand column
            product_id_col: Name of the product ID column (optional)
            
        Returns:
            Prepared DataFrame
        """
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values([product_id_col, date_col] if product_id_col else [date_col])
        
        # Fill missing values
        df[demand_col] = df[demand_col].fillna(0)
        
        # Add time-based features
        df = self.feature_engineering.add_time_features(df, date_col)
        
        # Add lag features
        if product_id_col:
            df = self.feature_engineering.add_lag_features(df, demand_col, product_id_col)
        else:
            df = self.feature_engineering.add_lag_features(df, demand_col)
        
        # Add rolling statistics
        df = self.feature_engineering.add_rolling_features(df, demand_col, product_id_col)
        
        # Add external data
        df = self.external_data.add_external_features(df, date_col)
        
        return df
    
    def train_multiple_models(self, df: pd.DataFrame, date_col: str, demand_col: str,
                            product_id: str = None, forecast_horizon: int = 30) -> Dict[str, Any]:
        """
        Train multiple forecasting models and compare performance
        
        Args:
            df: Prepared DataFrame
            date_col: Name of the date column
            demand_col: Name of the demand column
            product_id: Product ID (optional)
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Training results and model comparisons
        """
        results = {
            'models_trained': [],
            'best_model': None,
            'model_performances': {},
            'training_date': datetime.now().isoformat()
        }
        
        # Split data for validation
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        # Train Prophet model
        if PROPHET_AVAILABLE:
            try:
                prophet_result = self._train_prophet_model(train_df, val_df, date_col, demand_col)
                results['models_trained'].append('prophet')
                results['model_performances']['prophet'] = prophet_result
                self.models[f'prophet_{product_id}' if product_id else 'prophet'] = prophet_result['model']
            except Exception as e:
                logger.error(f"Prophet training failed: {str(e)}")
        
        # Train ARIMA model
        if STATSMODELS_AVAILABLE:
            try:
                arima_result = self._train_arima_model(train_df, val_df, date_col, demand_col)
                results['models_trained'].append('arima')
                results['model_performances']['arima'] = arima_result
                self.models[f'arima_{product_id}' if product_id else 'arima'] = arima_result['model']
            except Exception as e:
                logger.error(f"ARIMA training failed: {str(e)}")
        
        # Train LSTM model
        if TORCH_AVAILABLE:
            try:
                lstm_result = self._train_lstm_model(train_df, val_df, date_col, demand_col)
                results['models_trained'].append('lstm')
                results['model_performances']['lstm'] = lstm_result
                self.models[f'lstm_{product_id}' if product_id else 'lstm'] = lstm_result['model']
            except Exception as e:
                logger.error(f"LSTM training failed: {str(e)}")
        
        # Train simple seasonal naive model as baseline
        baseline_result = self._train_baseline_model(train_df, val_df, date_col, demand_col)
        results['models_trained'].append('baseline')
        results['model_performances']['baseline'] = baseline_result
        
        # Determine best model
        best_model_name = min(results['model_performances'].keys(), 
                            key=lambda x: results['model_performances'][x]['mae'])
        results['best_model'] = best_model_name
        
        self.model_performances[product_id if product_id else 'default'] = results
        
        return results
    
    def _train_prophet_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           date_col: str, demand_col: str) -> Dict[str, Any]:
        """Train Prophet model"""
        # Prepare data for Prophet
        prophet_df = train_df[[date_col, demand_col]].rename(columns={
            date_col: 'ds',
            demand_col: 'y'
        })
        
        # Create and train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # Add external regressors if available
        external_cols = [col for col in train_df.columns if col.startswith('external_')]
        for col in external_cols:
            model.add_regressor(col)
            prophet_df[col] = train_df[col]
        
        model.fit(prophet_df)
        
        # Make predictions on validation set
        val_prophet_df = val_df[[date_col, demand_col]].rename(columns={
            date_col: 'ds',
            demand_col: 'y'
        })
        
        for col in external_cols:
            val_prophet_df[col] = val_df[col]
        
        forecast = model.predict(val_prophet_df[['ds'] + external_cols])
        
        # Calculate metrics
        mae = mean_absolute_error(val_df[demand_col], forecast['yhat'])
        mse = mean_squared_error(val_df[demand_col], forecast['yhat'])
        rmse = np.sqrt(mse)
        
        return {
            'model': model,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'forecast': forecast
        }
    
    def _train_arima_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                         date_col: str, demand_col: str) -> Dict[str, Any]:
        """Train ARIMA model"""
        # Prepare time series
        ts = train_df.set_index(date_col)[demand_col]
        
        # Check stationarity
        def check_stationarity(timeseries):
            result = adfuller(timeseries)
            return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
        
        # Differencing if needed
        d = 0
        diff_series = ts.copy()
        while not check_stationarity(diff_series) and d < 2:
            d += 1
            diff_series = diff_series.diff().dropna()
        
        # Auto-select ARIMA parameters (simplified)
        # In practice, you'd use more sophisticated parameter selection
        p, q = 1, 1
        
        # Fit ARIMA model
        model = ARIMA(ts, order=(p, d, q))
        fitted_model = model.fit()
        
        # Make predictions
        forecast = fitted_model.forecast(steps=len(val_df))
        
        # Calculate metrics
        mae = mean_absolute_error(val_df[demand_col], forecast)
        mse = mean_squared_error(val_df[demand_col], forecast)
        rmse = np.sqrt(mse)
        
        return {
            'model': fitted_model,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'forecast': forecast
        }
    
    def _train_lstm_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        date_col: str, demand_col: str) -> Dict[str, Any]:
        """Train LSTM model"""
        # Prepare data for LSTM
        sequence_length = 30  # Look back 30 days
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:(i + seq_length)])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[[demand_col]])
        val_scaled = scaler.transform(val_df[[demand_col]])
        
        # Create sequences
        X_train, y_train = create_sequences(train_scaled.flatten(), sequence_length)
        X_val, y_val = create_sequences(val_scaled.flatten(), sequence_length)
        
        # Create LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(torch.FloatTensor(X_train).unsqueeze(-1))
            loss = criterion(outputs, torch.FloatTensor(y_train).unsqueeze(-1))
            loss.backward()
            optimizer.step()
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(torch.FloatTensor(X_val).unsqueeze(-1))
            predictions = scaler.inverse_transform(predictions.numpy())
        
        # Calculate metrics
        actual = val_df[demand_col].values[sequence_length:]
        mae = mean_absolute_error(actual, predictions.flatten())
        mse = mean_squared_error(actual, predictions.flatten())
        rmse = np.sqrt(mse)
        
        return {
            'model': model,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'forecast': predictions.flatten()
        }
    
    def _train_baseline_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                            date_col: str, demand_col: str) -> Dict[str, Any]:
        """Train baseline seasonal naive model"""
        # Use seasonal naive approach (same day of week from previous week)
        train_df['day_of_week'] = pd.to_datetime(train_df[date_col]).dt.dayofweek
        
        # Calculate seasonal means
        seasonal_means = train_df.groupby('day_of_week')[demand_col].mean()
        
        # Make predictions
        val_df['day_of_week'] = pd.to_datetime(val_df[date_col]).dt.dayofweek
        predictions = val_df['day_of_week'].map(seasonal_means)
        
        # Calculate metrics
        mae = mean_absolute_error(val_df[demand_col], predictions)
        mse = mean_squared_error(val_df[demand_col], predictions)
        rmse = np.sqrt(mse)
        
        return {
            'model': seasonal_means,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'forecast': predictions
        }
    
    def forecast_demand(self, product_id: str, forecast_horizon: int = 30,
                       confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Generate demand forecast for a product
        
        Args:
            product_id: Product identifier
            forecast_horizon: Number of periods to forecast
            confidence_interval: Confidence interval for predictions
            
        Returns:
            Forecast results
        """
        model_key = f'prophet_{product_id}' if f'prophet_{product_id}' in self.models else 'prophet'
        
        if model_key not in self.models:
            # Create a synthetic forecast for products without trained models
            logger.warning(f"No trained model found for product {product_id}, generating synthetic forecast")
            return self._generate_synthetic_forecast(product_id, forecast_horizon, confidence_interval)
        
        model = self.models[model_key]
        
        # Generate future dates
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=forecast_horizon,
            freq='D'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Add external features if available
        future_df = self.external_data.add_external_features(future_df, 'ds')
        
        # Make forecast
        if isinstance(model, Prophet):
            forecast = model.predict(future_df)
            
            results = {
                'dates': future_dates.tolist(),
                'forecast': forecast['yhat'].tolist(),
                'lower_bound': forecast['yhat_lower'].tolist(),
                'upper_bound': forecast['yhat_upper'].tolist(),
                'confidence_interval': confidence_interval,
                'model_type': 'prophet'
            }
        else:
            # Handle other model types
            forecast = model.forecast(steps=forecast_horizon)
            
            results = {
                'dates': future_dates.tolist(),
                'forecast': forecast.tolist() if hasattr(forecast, 'tolist') else [forecast],
                'lower_bound': None,
                'upper_bound': None,
                'confidence_interval': confidence_interval,
                'model_type': 'other'
            }
        
        # Add production recommendations
        results['production_recommendations'] = self._generate_production_recommendations(
            results['forecast']
        )
        
        return results
    
    def _generate_production_recommendations(self, forecast: List[float]) -> Dict[str, Any]:
        """Generate production recommendations based on forecast"""
        total_demand = sum(forecast)
        avg_daily_demand = np.mean(forecast)
        peak_demand = max(forecast)
        
        # Safety stock calculation (simplified)
        safety_stock = avg_daily_demand * 0.2  # 20% safety stock
        
        recommendations = {
            'total_forecasted_demand': total_demand,
            'average_daily_demand': avg_daily_demand,
            'peak_demand': peak_demand,
            'recommended_production': total_demand + safety_stock,
            'safety_stock': safety_stock,
            'production_schedule': [],
            'inventory_alerts': []
        }
        
        # Generate production schedule
        for i, daily_demand in enumerate(forecast):
            production_qty = daily_demand + (safety_stock / len(forecast))
            recommendations['production_schedule'].append({
                'day': i + 1,
                'forecasted_demand': daily_demand,
                'recommended_production': production_qty
            })
        
        # Generate alerts
        if peak_demand > avg_daily_demand * 1.5:
            recommendations['inventory_alerts'].append({
                'type': 'high_demand_alert',
                'message': f"Peak demand ({peak_demand:.1f}) significantly higher than average"
            })
        
        return recommendations
    
    def get_model_performance(self, product_id: str = None) -> Dict[str, Any]:
        """Get model performance metrics"""
        key = product_id if product_id else 'default'
        return self.model_performances.get(key, {})
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'model_performances': self.model_performances,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.model_performances = model_data['model_performances']
        logger.info(f"Models loaded from {filepath}")

    def _generate_synthetic_forecast(self, product_id: str, forecast_horizon: int = 30,
                                       confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Generate synthetic demand forecast for products without trained models
        
        Args:
            product_id: Product identifier
            forecast_horizon: Number of periods to forecast
            confidence_interval: Confidence interval for predictions
            
        Returns:
            Synthetic forecast results
        """
        # Generate future dates
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=forecast_horizon,
            freq='D'
        )
        
        # Create synthetic demand based on product characteristics
        base_demand = self._estimate_base_demand(product_id)
        
        # Add seasonality and trends
        synthetic_forecast = []
        for i, date in enumerate(future_dates):
            # Add weekly seasonality (lower on weekends)
            day_of_week = date.weekday()
            weekly_factor = 0.8 if day_of_week >= 5 else 1.0
            
            # Add monthly seasonality
            monthly_factor = 1.0 + 0.2 * np.sin(2 * np.pi * date.day / 30)
            
            # Add some noise and trend
            noise = np.random.normal(0, 0.1)
            trend = 0.001 * i  # Small positive trend
            
            demand = base_demand * weekly_factor * monthly_factor * (1 + trend + noise)
            demand = max(0, demand)  # Ensure non-negative
            synthetic_forecast.append(demand)
        
        # Calculate confidence bounds
        forecast_array = np.array(synthetic_forecast)
        std_dev = np.std(forecast_array)
        confidence_multiplier = 1.96 if confidence_interval == 0.95 else 1.645
        
        lower_bound = forecast_array - confidence_multiplier * std_dev
        upper_bound = forecast_array + confidence_multiplier * std_dev
        
        results = {
            'dates': future_dates.tolist(),
            'forecast': synthetic_forecast,
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'confidence_interval': confidence_interval,
            'model_type': 'synthetic'
        }
        
        # Add production recommendations
        results['production_recommendations'] = self._generate_production_recommendations(
            results['forecast']
        )
        
        return results
    
    def _estimate_base_demand(self, product_id: str) -> float:
        """
        Estimate base demand for a product without historical data
        
        Args:
            product_id: Product identifier
            
        Returns:
            Estimated base demand
        """
        # Simple heuristic based on product_id
        # In a real system, this would use product category, price, etc.
        product_hash = hash(product_id) % 1000
        base_demand = 10 + (product_hash % 50)  # Between 10 and 60
        
        return float(base_demand)

class FeatureEngineer:
    """Feature engineering for demand forecasting"""
    
    def add_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        df['year'] = pd.to_datetime(df[date_col]).dt.year
        df['month'] = pd.to_datetime(df[date_col]).dt.month
        df['day'] = pd.to_datetime(df[date_col]).dt.day
        df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek
        df['day_of_year'] = pd.to_datetime(df[date_col]).dt.dayofyear
        df['week_of_year'] = pd.to_datetime(df[date_col]).dt.isocalendar().week
        df['quarter'] = pd.to_datetime(df[date_col]).dt.quarter
        
        # Holiday indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = pd.to_datetime(df[date_col]).dt.is_month_start.astype(int)
        df['is_month_end'] = pd.to_datetime(df[date_col]).dt.is_month_end.astype(int)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, target_col: str, 
                        group_col: str = None, lags: List[int] = None) -> pd.DataFrame:
        """Add lag features"""
        if lags is None:
            lags = [1, 7, 14, 30]
        
        df = df.copy()
        
        for lag in lags:
            if group_col:
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
            else:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, target_col: str,
                           group_col: str = None, windows: List[int] = None) -> pd.DataFrame:
        """Add rolling window features"""
        if windows is None:
            windows = [7, 14, 30]
        
        df = df.copy()
        
        for window in windows:
            if group_col:
                df[f'{target_col}_rolling_mean_{window}'] = df.groupby(group_col)[target_col].rolling(window=window).mean().reset_index(0, drop=True)
                df[f'{target_col}_rolling_std_{window}'] = df.groupby(group_col)[target_col].rolling(window=window).std().reset_index(0, drop=True)
            else:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        
        return df

class ExternalDataCollector:
    """Collect external data for demand forecasting"""
    
    def __init__(self):
        self.weather_data = None
        self.economic_data = None
        self.social_media_data = None
    
    def add_external_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add external features to the dataframe"""
        df = df.copy()
        
        # Add weather data (placeholder)
        df['temperature'] = np.random.normal(20, 5, len(df))  # Mock temperature
        df['precipitation'] = np.random.exponential(2, len(df))  # Mock precipitation
        
        # Add economic indicators (placeholder)
        df['economic_index'] = np.random.normal(100, 10, len(df))  # Mock economic index
        
        # Add social media sentiment (placeholder)
        df['social_sentiment'] = np.random.uniform(-1, 1, len(df))  # Mock sentiment
        
        return df
    
    def fetch_weather_data(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weather data (placeholder)"""
        # In practice, this would connect to weather APIs
        return pd.DataFrame()
    
    def fetch_economic_data(self, indicators: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch economic indicators (placeholder)"""
        # In practice, this would connect to economic data APIs
        return pd.DataFrame()
    
    def fetch_social_media_sentiment(self, keywords: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch social media sentiment (placeholder)"""
        # In practice, this would connect to social media APIs
        return pd.DataFrame()

class DemandAnomalyDetector:
    """Detect anomalies in demand patterns"""
    
    def __init__(self):
        self.threshold_multiplier = 2.0
    
    def detect_anomalies(self, df: pd.DataFrame, demand_col: str,
                        date_col: str = None) -> pd.DataFrame:
        """
        Detect anomalies in demand data
        
        Args:
            df: DataFrame with demand data
            demand_col: Name of demand column
            date_col: Name of date column (optional)
            
        Returns:
            DataFrame with anomaly flags
        """
        df = df.copy()
        
        # Calculate rolling statistics
        df['rolling_mean'] = df[demand_col].rolling(window=30, min_periods=1).mean()
        df['rolling_std'] = df[demand_col].rolling(window=30, min_periods=1).std()
        
        # Calculate z-score
        df['z_score'] = (df[demand_col] - df['rolling_mean']) / df['rolling_std']
        
        # Flag anomalies
        df['is_anomaly'] = np.abs(df['z_score']) > self.threshold_multiplier
        df['anomaly_type'] = np.where(df['z_score'] > self.threshold_multiplier, 'high',
                                     np.where(df['z_score'] < -self.threshold_multiplier, 'low', 'normal'))
        
        # Add anomaly score
        df['anomaly_score'] = np.abs(df['z_score']) / self.threshold_multiplier
        
        return df
    
    def get_anomaly_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get insights about detected anomalies"""
        anomalies = df[df['is_anomaly']]
        
        insights = {
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df),
            'high_anomalies': len(anomalies[anomalies['anomaly_type'] == 'high']),
            'low_anomalies': len(anomalies[anomalies['anomaly_type'] == 'low']),
            'avg_anomaly_score': anomalies['anomaly_score'].mean() if len(anomalies) > 0 else 0,
            'anomaly_dates': anomalies.index.tolist() if hasattr(anomalies.index, 'tolist') else []
        }
        
        return insights
