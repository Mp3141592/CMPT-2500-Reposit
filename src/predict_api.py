from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

def predict(stock_type, mileage, msrp, model_year, make,
    transmission_from_vin):
    """
    Predicts the price of a car based on features given.
    """

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

    makedata = pd.DataFrame({f'make_{make}'})

    # 
    if 'transmission_from_vin_A' in new_data.columns:
        new_data = new_data.rename(columns={'transmission_from_vin_A': "transmission_from_vin_M"})
    
    for i in make_list:
        if i not in makedata.columns:
            new_data[i] = 0
        else:
            new_data[i] = 1
    
    return new_data



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
            "/v1/predict1": "Uses a model to predict info in json format",
            "/v2/predict1": "Same as v1 but different model"
        },

        "input_format" : {
            "stock_type": "Must be a string and either USED or NEW",
            "mileage": "Float with 1 decimal",
            "msrp": "Int with no deciamals",
            "model_year": "Int of 4 numbers and stays below 2025",
            "make": "String that starts with a capital",
            "transmission_from_vin": "String that is either A for Auto or M for Manual"
        },
        "example_request": {
            "stock_type": "USED",
            "mileage": 543.0,
            "msrp": 20,
            "model_year": 2023,
            "make": "Volvo",
            "transmission_from_vin": "M"          
        },
        "example_response": {
            "sucess": True,
            "prediction":{
                "price": 45340      
            }
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




if __name__ == "__main__":
    app.run(host='127.0.0.1', port=9999, debug=True)