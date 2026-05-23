FROM python:3.11-slim

WORKDIR /

COPY . /app

RUN pip install -r /app/requirements.txt

CMD ["uvicorn", "--host=0.0.0.0", "app.main:app"]
