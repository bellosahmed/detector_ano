# Anomaly Detector — one-command local web app.
#   docker build -t anomaly-detector .
#   docker run -p 3020:3020 anomaly-detector   →   http://localhost:3020
FROM python:3.12-slim

WORKDIR /app

COPY . .
# install the package (pulls its declared dependencies) + the web entry point
RUN pip install --no-cache-dir .

EXPOSE 3020
# bind to 0.0.0.0 so the app is reachable from outside the container
ENV HOST=0.0.0.0
CMD ["anomaly-detector-web"]
