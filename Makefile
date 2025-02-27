PIP := .venv/bin/pip

init:
	pyhon3 -m venv .venv
	$(source) .venv/bin/activate 
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

preprocess:
	$(PYTHON) src/preprocess.py

train:
	$(PYTHON) src/train.py

eval:
	$(PYTHON) src/evaluate.py