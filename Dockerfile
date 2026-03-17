FROM python:3.12.13-bookworm

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY power_forecast /power_forecast

CMD uvicorn power_forecast.api.fast:app --host 0.0.0.0 --port $PORT
