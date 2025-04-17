from flask import Flask, jsonify, request
import pandas as pd
import joblib
import logging
import time
import threading
import psutil
import os

# Import Flask and Prometheus packages
from prometheus_client import Counter, Histogram, Gauge
from prometheus_flask_exporter import PrometheusMetrics

# Initialize Flask app
app = Flask(__name__)

# Initialize PrometheusMetrics with default registry for compatibility with Grafana dashboard
metrics = PrometheusMetrics(app)

# Add basic app info metric
metrics.info('flask_app_info', 'Application info', version='1.0.0')
    
# Register prediction metrics with names matching Grafana dashboard expectations
prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests', ['model_version'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_version'])
request_counter = Counter("api_requests_total", "Total number of API requests", ["method", "endpoint"])

# Use standard names for memory and CPU metrics that Grafana dashboard is configured to use
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('app.log')  # Log to file            
    ]
)

logger = logging.getLogger(__name__)

def predict(stock_type, mileage, msrp, model_year, make,
    transmission_from_vin, model):
    """
    Predicts the price of a car based on features given.
    """
    try:

        logger.info(f"Received prediction request with stock_type: {stock_type}")
        logger.info(f"Received prediction request with mileage: {mileage}")
        logger.info(f"Received prediction request with msrp: {msrp}")
        logger.info(f"Received prediction request with model_year: {model_year}")
        logger.info(f"Received prediction request with make: {make}")
        logger.info(f"Received prediction request with transmission: {transmission_from_vin}")
        logger.info(f"Received prediction request with model: {model}")

        # List of all makes in model
        make_list = ['make_Acura', 'make_Alfa Romeo', 'make_Audi',
        'make_BMW', 'make_Buick', 'make_Cadillac', 'make_Chevrolet',
        'make_Chrysler', 'make_Dodge', 'make_Fiat', 'make_Ford', 'make_GMC',
        'make_Genesis', 'make_Honda', 'make_Hyundai', 'make_Infiniti',
        'make_Jaguar', 'make_Jeep', 'make_Kia', 'make_Land Rover', 'make_Lexus',
        'make_Lincoln', 'make_Maserati', 'make_Mazda', 'make_Mercedes-Benz',
        'make_Mini', 'make_Mitsubishi', 'make_Nissan', 'make_Polestar',
        'make_Pontiac', 'make_Porsche', 'make_Ram', 'make_Rivian', 'make_Scion',
        'make_Smart', 'make_Subaru', 'make_Suzuki', 'make_Tesla', 'make_Toyota',
        'make_Volkswagen', 'make_Volvo']
        
        # Create basic dataframe from new data
        d = {'stock_type': [stock_type],
            'mileage': [mileage],
            'model_year': [model_year],
            'msrp': [msrp],
            'transmission_from_vin': [transmission_from_vin]
            }
        new_data = pd.DataFrame(d)

        # Encode new data
        new_data = pd.get_dummies(new_data,prefix=['transmission_from_vin'], columns = ['transmission_from_vin'], dtype=float)
        new_data = pd.get_dummies(new_data,prefix=['stock_type'], columns = ['stock_type'], dtype=float)

        # Grab make data
        makedata = pd.DataFrame({f'make_{make}'})

        # Rename if transmission column name different
        if 'transmission_from_vin_A' in new_data.columns:
            new_data = new_data.rename(columns={'transmission_from_vin_A': "transmission_from_vin_M"})
        
        # Add in car make columns
        for i in make_list:
            if i not in makedata.columns:
                new_data[i] = 0
            else:
                new_data[i] = 1

        price = model.predict(new_data)
        
        logger.info(f"Prediction successful")

        return float(price)
    
    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}") 
        raise 

# Decorater for app (endpoint)
@app.route('/Car_Price_Prediction_home', methods=['GET'])
def home():
    app_info= {
        "name": "Car_Price_Prediction_API",
        "description": "This API takes features from cars and predicts a reasonable price for them",
        "version": "v.1.0",
        "endpoints": {
            "/Car_Price_Prediction_home": "The home page",
            "/health_status": "Indicates if API is available and ready",
            "/v1/predict1": "Uses v1 model to predict price",
            "/v2/predict1": "Uses v2 model to predict price"
        },

        "input_format" : {
            "stock_type": "Must be a string and either USED or NEW",
            "mileage": "Float with 1 decimal",
            "msrp": "Int with no deciamals",
            "model_year": "Int of 4 numbers",
            "make": "String that starts with a capital",
            "transmission_from_vin": "String that is either A for Auto or M for Manual"
        },
        "example_request": {
            "stock_type": "USED",
            "mileage": 543.0,
            "msrp": 20,
            "model_year": 2023,
            "make": "Alfa Romeo",
            "transmission_from_vin": "M"          
        },
        "example_response": {
            "success": True,  
            "predicted_price": 45340      
            }
        }
    
    return jsonify(app_info)

    
@app.route('/health_status', methods=['GET'])
def health_check():

    health = {
        'status': "UP",
        "message": "Car Price Prediction is up"
    }
    return jsonify(health)

@app.route('/v1/predict', methods=['POST'])
def v1():

    start_time = time.time()
    
    request_counter.labels(method="POST", endpoint="/v1/predict").inc()

    if not request.is_json:
        return jsonify({"error": "Request must be JSON data"})
    
    data = request.json
    
    prediction_requests.labels(model_version="v1").inc()

    if "stock_type" not in data:
        return jsonify ({"error": "Missing stock_type data"})
    
    if "mileage" not in data:
        return jsonify ({"error": "Missing mileage data"})
    
    if "msrp" not in data:
        return jsonify ({"error": "Missing msrp data"})
    
    if "model_year" not in data:
        return jsonify ({"error": "Missing model_year data"})
    
    if "make" not in data:
        return jsonify ({"error": "Missing make data"})
    
    if "transmission_from_vin" not in data:
        return jsonify ({"error": "Missing transmission_from_vin data"})
    
    stock_type = data.get('stock_type')
    mileage = data.get('mileage')
    msrp = data.get('msrp')
    model_year = data.get('model_year')
    make = data.get('make')
    transmission_from_vin = data.get('transmission_from_vin')

    model = joblib.load('/app/models/ridge_model_v1.jlib')

    results = predict(stock_type, mileage, msrp, model_year, make, transmission_from_vin, model)
    prediction_time.labels(model_version="v1").observe(time.time() - start_time)

    return jsonify({
        "success": True,
        "price_predicted": results
    })
    



@app.route('/v2/predict', methods=['POST'])
def v2():

    start_time = time.time()
    
    request_counter.labels(method="POST", endpoint="/v2/predict").inc()
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON data"})
    
    data = request.json
    prediction_requests.labels(model_version="v2").inc()
    
    if "stock_type" not in data:
        return jsonify ({"error": "Missing stock_type data"})
    
    if "mileage" not in data:
        return jsonify ({"error": "Missing mileage data"})
    
    if "msrp" not in data:
        return jsonify ({"error": "Missing msrp data"})
    
    if "model_year" not in data:
        return jsonify ({"error": "Missing model_year data"})
    
    if "make" not in data:
        return jsonify ({"error": "Missing make data"})
    
    if "transmission_from_vin" not in data:
        return jsonify ({"error": "Missing transmission_from_vin data"})
    
    stock_type = data.get('stock_type')
    mileage = data.get('mileage')
    msrp = data.get('msrp')
    model_year = data.get('model_year')
    make = data.get('make')
    transmission_from_vin = data.get('transmission_from_vin')

    model = joblib.load('/app/models/ridge_model_v2.jlib')

    results = predict(stock_type, mileage, msrp, model_year, make, transmission_from_vin, model)

    prediction_time.labels(model_version="v2").observe(time.time() - start_time)

    return jsonify({
        "success": True,
        "price_predicted": results
    })

def monitor_resources():
    """Background thread function to monitor resource usage"""
    while True:
        try:
            # Update memory and CPU metrics
            process = psutil.Process(os.getpid())
            
            # Set memory usage in bytes (Resident Set Size)
            memory_usage.set(process.memory_info().rss)  
            
            # Set CPU percentage - this is what the grafana dashboard expects
            cpu_usage.set(process.cpu_percent(interval=1.0))
            
            # Log successful metric updates at regular intervals
            logger.debug(f"Updated memory usage: {process.memory_info().rss} bytes, CPU usage: {process.cpu_percent()}%")
            
            # Wait before next update
            time.sleep(15)  # Update metrics every 15 seconds
        except Exception as e:
            logger.error(f"Error in resource monitoring thread: {e}")
            time.sleep(60)

if __name__ == "__main__":
    # Start the resource monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # Set host and port
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = 9000
    
    # Log that we're starting with metrics enabled
    logger.info(f"Starting Flask app on {host}:{port} with Prometheus metrics enabled at /metrics")
    
    # Run the app
    app.run(host=host, port=port, debug=False)