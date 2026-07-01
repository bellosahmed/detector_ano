# Common tasks. Run `make help` to list them.
PY ?= python3.12

.PHONY: help install test verify samples benchmarks web docker

help:
	@echo "install     - pip install the package (editable)"
	@echo "test        - run the test suite"
	@echo "verify      - run the verification gate (query==pandas + synthetic P/R/F1)"
	@echo "samples     - regenerate the sample datasets + answer keys"
	@echo "benchmarks  - run the benchmarks, writing benchmarks/results/*.txt"
	@echo "web         - launch the web UI at http://localhost:3020"
	@echo "docker      - build and run the web UI in Docker"

install:
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

verify:
	$(PY) scripts/verify.py

samples:
	$(PY) scripts/make_samples.py

benchmarks:
	bash scripts/run_benchmarks.sh

web:
	$(PY) -m anomaly_detector.web.app

docker:
	docker build -t anomaly-detector . && docker run -p 3020:3020 anomaly-detector
