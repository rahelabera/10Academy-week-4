FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/api /app/src/api
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]