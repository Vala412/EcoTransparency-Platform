"""
Logging configuration for the EcoTransparency Platform
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "ecotransparency", log_level: str = "INFO", 
                log_file: str = None) -> logging.Logger:
    """
    Set up application logger with file and console handlers
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls
    
    Args:
        logger: Logger instance
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Calling function: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {str(e)}")
                raise
        return wrapper
    return decorator

def log_api_request(logger: logging.Logger):
    """
    Decorator to log API requests
    
    Args:
        logger: Logger instance
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"API request: {func.__name__}")
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"API request {func.__name__} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"API request {func.__name__} failed after {duration:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator

class StructuredLogger:
    """
    Structured logging class for consistent log formatting
    """
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = setup_logger(name, log_level)
    
    def log_event(self, event_type: str, message: str, **kwargs):
        """
        Log structured event
        
        Args:
            event_type: Type of event (e.g., 'model_training', 'api_request')
            message: Log message
            **kwargs: Additional structured data
        """
        log_data = {
            'event_type': event_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        self.logger.info(f"[{event_type.upper()}] {message} | Data: {log_data}")
    
    def log_model_performance(self, model_name: str, metrics: dict, **kwargs):
        """
        Log model performance metrics
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
            **kwargs: Additional context
        """
        self.log_event(
            'model_performance',
            f"Model {model_name} performance recorded",
            model_name=model_name,
            metrics=metrics,
            **kwargs
        )
    
    def log_data_processing(self, operation: str, records_processed: int, **kwargs):
        """
        Log data processing operations
        
        Args:
            operation: Type of operation
            records_processed: Number of records processed
            **kwargs: Additional context
        """
        self.log_event(
            'data_processing',
            f"Data processing: {operation}",
            operation=operation,
            records_processed=records_processed,
            **kwargs
        )
    
    def log_user_activity(self, user_id: str, activity: str, **kwargs):
        """
        Log user activity
        
        Args:
            user_id: User identifier
            activity: Activity description
            **kwargs: Additional context
        """
        self.log_event(
            'user_activity',
            f"User {user_id}: {activity}",
            user_id=user_id,
            activity=activity,
            **kwargs
        )
    
    def log_error(self, error_type: str, error_message: str, **kwargs):
        """
        Log error with structured format
        
        Args:
            error_type: Type of error
            error_message: Error message
            **kwargs: Additional context
        """
        self.log_event(
            'error',
            f"Error occurred: {error_type}",
            error_type=error_type,
            error_message=error_message,
            **kwargs
        )
        
        # Also log to error level
        self.logger.error(f"[ERROR] {error_type}: {error_message}")

# Global logger instance
logger = setup_logger()
