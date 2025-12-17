FROM python:3.10-slim

WORKDIR /app

# Системные зависимости для HDF5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY app/ ./
COPY models/ ./models/

# Проверка файлов
RUN echo "=== Директория модели ===" && ls -la models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]