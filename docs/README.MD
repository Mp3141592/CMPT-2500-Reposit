Makefile should allow you to run everything for the data.

Steps for model training process:

1. Raw data is taken from the raw data folder in data.
2. Data is processed in preprocess.py located under src, and then split and exported as 4 filese to processed.
3. Data is exported to train.py where it is encoded and trained on a ridge regression, which then has the model saved to models.
4. model brought to evalulate.py and evaluated on X_teste and y_test.

make init
make preprocess
make train
make eval