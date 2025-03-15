# Overview of predict_api.py 

## What does it do?
This api predicts how much a car should be priced at by taking car information in the form of a json request and using that information to predict its price.

## How to use it?
To use this api, after confirming that it is running by going to http://127.0.0.1:9999/health_status and seeing that status is up message, use the format and code below to enter the cars information and then run it in the terminal.

curl -X POST http://127.0.0.1:9999/<v1 or v2>/predict \\\
-H "Content-Type: application/json" \\\
-d '{
    "stock_type": "Input USED or NEW",\
    "mileage": input a number with 1 decimal,\
    "msrp": input a number,\
    "model_year": input a full year,
    "make": "input the make with a capital ",\
    "transmission_from_vin": "Input M for Manual or A for Auto"\
}'

For more information or examples, go to 
http://127.0.0.1:9999/Car_Price_Prediction_home

## What should I expect to get as output?
Your output should look something like this in your terminal

        "success": True,\
        "price_predicted": 25403.99

## Endpoints
There are four endpoints included in this api.

/Car_Price_Prediction_home : This is the home page of the api. It has details such as the version and name of api, the other endpoints avaliable for use, the proper input format to predict cars with, examples of what inputs should look like, and examples of what outputs should look like.


/health_status: Health status tells the user if the app is up and ready to be used, and will return a json formatted msg like the following.

        'status': "UP",
        "message": "Car Price Prediction is up"


/v1/predict: V1 is the endpoint used to predict prices using the v1 model. A successful prediction will return the following

        "success": True,
        "price_predicted": results (will be a number)

/v2/predict: V2 is the endpoint used to predict prices using the v2 model. It will return something in the same format as v1.

## Parameters

stock_type: This is the stock type of the car, so the field should either be USED or NEW.

mileage: This is the mileage of the car and is a number with  one decimal place.

msrp: This is the price the manufacter recommends selling the car for. This will be a number with no decimals.

model_year: Model year is the year the model was created. It will be a four digit number.

make: This is the make of the car. It will be a string like Volvo or Alfa_Romeo

transmission_from_vin: This is the transmission of the car. It will be a string that is either A for automatic or M for manual.

## Request Body Format
The request must be in json format and should be something like the below.

    {"stock_type": "USED", "mileage": 543.0, "msrp": 2000, "make": "Volvo", "transmission_from_vin": "M"}