# Swap from 'slim' to the full Python image to ensure all build tools are present
FROM python:3.10

WORKDIR /app

COPY requirements.txt .

# Upgrade pip first, then install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn torch torchvision torchaudio mlflow
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]