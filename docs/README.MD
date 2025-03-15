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