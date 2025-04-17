This is the git repository for the CMPT 2500 lab assignments, though it is using the CMPT 3830 project.

Steps for model training process:

1. Raw data is taken from the raw data folder in data.
2. Data is processed in preprocess.py located under src, and then split and exported as 4 filese to processed.
3. Data is exported to train.py where it is encoded and trained on a ridge regression, which then has the model saved to models.
4. model brought to evalulate.py and evaluated on X_teste and y_test.

## Makefile commands in terminal
Makefile should allow you to run everything for the data.

Makefile processes:
make init
make preprocess
make train
make eval

Run make init to install requirements, source .venv/bin/activate to activate the virtual environment, and make preprocess to clean, train, and evaluate data.

Currently,  preprocess is the only one that needs to be run out of it, train, and eval. Both Train and eval have been turned
into modules that are currently called at preprocess (though this may change to a proper main file later on)

## Docker Contanerization

To build the docker images, use the commmand docker-compose up --build in the terminal.

To use the flask application in the container, run the container and then open a new termianl. Afterwards, use docker-compose run app python src/predict_api.py in the terminal and enter a similar format like how you would do so for the regular flask application (see API.MD for more information)

## Prometheus and Graphana

Preometheus is used to collect metrics and alerts about the app deployed and any time its run. This data is then sent to Graphana and placed into a dashboard so we can see the statistics of how our app is running or how the model is doing (such as cpu usage and r2 score.)

Prometheus has a 4 alerts:
1. High Error Rate: Give warning if predction error rate is greater than 10%.
2. Slow Prediction Response: Give a warning if the model is predicting slower than usual.
3. High Memory Usage: Give a warning if the memory exceeds 1.5GB of usage.
4. High Request Volume: API request rate is more than 10 in a minute.

This is the link to Prometheus: http://localhost:9090/graph?g0.expr=&g0.tab=1&g0.display_mode=lines&g0.show_exemplars=0&g0.range_input=1h

\
Graphana has two dashboards. 
1. Car Price API: Shows the statistics of the API (such as response time, memory usage, prediction time, etc.)
2. Model Training Dashboard: Dashboard should show the metrics of the trained model or model in training (such as epochs used, accurracy, loss, etc) but is currently not working.

This is the link to the dashboards: http://localhost:3000/?orgId=1&from=now-6h&to=now&timezone=browser

\
Use the command below a few times to stress test the API and see the alerts and information provided on both Prometheus and Graphana.
\
\
seq 1 1000 | xargs -n1 -P50 -I{} curl -s -X POST http://127.0.0.1:9000/v1/predict -H "Content-Type: application/json" -d '{
    "stock_type": "USED",
    "mileage": 45782.3,
    "msrp": 23000,
    "model_year": 2018,
    "make": "Toyota",
    "transmission_from_vin": "A"
}' 
